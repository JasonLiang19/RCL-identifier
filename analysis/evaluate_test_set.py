#!/usr/bin/env python3
"""
Evaluate all 6 trained models on the held-out test set.
Generates a comparison table of performance metrics.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.data_loader import read_csv_with_annotations
from data.encoders import OneHotEncoder, BLOSUMEncoder, ESM2_650M_Encoder, EncodedDataset
from models import get_model
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score


def load_model_and_encoder(model_name, device):
    """Load trained model and appropriate encoder."""
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'analysis' / 'results' / model_name
    
    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Determine encoding and dimension
    if 'onehot' in model_name:
        encoding_type = 'onehot'
        encoding_dim = 21
        max_length = config['data']['max_length']
        encoder = OneHotEncoder(max_length=max_length)
    elif 'blosum' in model_name:
        encoding_type = 'blosum'
        encoding_dim = 20
        max_length = config['data']['max_length']
        encoder = BLOSUMEncoder(max_length=max_length)
    elif 'esm2_650m' in model_name:
        encoding_type = 'esm2_650m'
        encoding_dim = 1280
        max_length = config['data']['max_length']
        encoder = ESM2_650M_Encoder(max_length=max_length, device=device)
    else:
        raise ValueError(f"Unknown encoding type: {model_name}")
    
    # Create model
    if 'cnn' in model_name:
        model_type = 'cnn'
    elif 'unet' in model_name:
        model_type = 'unet'
    elif 'lstm' in model_name:
        model_type = 'lstm'
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    model = get_model(
        model_type,
        encoding_dim,
        config['models'][model_type],
        num_classes=2
    )
    
    # Load weights
    checkpoint = torch.load(model_dir / 'best_model.pt', map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, encoder, encoding_type, max_length


def load_test_data(max_length=1024):
    """Load test set from Uniprot_Test_Set.csv."""
    test_file = Path(__file__).parent.parent / 'data' / 'raw' / 'Uniprot_Test_Set.csv'
    
    # Read test data
    ids, labels, sequences, rcl_sequences = read_csv_with_annotations(
        test_file,
        max_length=max_length
    )
    
    return ids, sequences, labels


def evaluate_model(model, encoder, test_ids, test_sequences, test_labels, device, batch_size=8):
    """Evaluate model on test set."""
    print(f"  Encoding {len(test_sequences)} test sequences...")
    
    # Encode sequences
    if isinstance(encoder, ESM2_650M_Encoder):
        encodings = encoder.encode_batch(test_sequences, batch_size=1)
    else:
        encodings = encoder.encode_batch(test_sequences)
    
    # Create dataset
    dataset = EncodedDataset(
        encodings,
        test_labels,
        ids=test_ids,
        sequences=test_sequences,
        lazy_load=False
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Make predictions
    print(f"  Making predictions...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='  Evaluating'):
            encodings_batch = batch['encoding'].to(device)
            labels_batch = batch['label'].to(device)
            
            # Forward pass
            outputs = model(encodings_batch)
            probs = torch.softmax(outputs, dim=-1)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)  # (N, max_length, 2)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, max_length, 2)
    
    # Calculate metrics (per-residue)
    # Flatten predictions and labels
    pred_flat = all_preds.reshape(-1, 2)  # (N*max_length, 2)
    label_flat = all_labels.reshape(-1, 2)  # (N*max_length, 2)
    
    # Get class predictions and true classes
    pred_classes = np.argmax(pred_flat, axis=1)
    true_classes = np.argmax(label_flat, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_classes, pred_classes, average='binary', pos_label=1
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        true_classes, pred_classes, average='macro'
    )
    mcc = matthews_corrcoef(true_classes, pred_classes)
    
    # AUC (using probability of RCL class)
    try:
        auc = roc_auc_score(true_classes, pred_flat[:, 1])
    except:
        auc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1_macro,
        'mcc': mcc,
        'auc': auc
    }
    
    return metrics


def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Models to evaluate
    models = [
        'onehot_cnn',
        'onehot_unet',
        'blosum_cnn',
        'blosum_unet',
        'esm2_650m_cnn',
        'esm2_650m_unet'
    ]
    
    # Load test data
    print("Loading test set...")
    test_ids, test_sequences, test_labels = load_test_data()
    print(f"✓ Loaded {len(test_sequences)} test sequences\n")
    
    # Evaluate each model
    results = []
    
    for model_name in models:
        print("="*70)
        print(f"Evaluating: {model_name}")
        print("="*70)
        
        try:
            # Load model and encoder
            model, encoder, encoding_type, max_length = load_model_and_encoder(model_name, device)
            
            # Evaluate
            metrics = evaluate_model(
                model, encoder, test_ids, test_sequences, test_labels, device
            )
            
            # Store results
            encoding_name = model_name.rsplit('_', 1)[0]
            model_type = model_name.rsplit('_', 1)[1]
            
            results.append({
                'Encoding': encoding_name,
                'Model': model_type.upper(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'F1-Macro': metrics['f1_macro'],
                'MCC': metrics['mcc'],
                'AUC': metrics['auc']
            })
            
            print(f"✓ Completed: {model_name}")
            print()
            
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            print()
    
    # Create results table
    df = pd.DataFrame(results)
    
    # Sort by encoding and model
    encoding_order = {'onehot': 0, 'blosum': 1, 'esm2_650m': 2}
    df['enc_order'] = df['Encoding'].map(encoding_order)
    df = df.sort_values(['enc_order', 'Model']).drop('enc_order', axis=1)
    
    # Print results
    print("\n" + "="*100)
    print("TEST SET EVALUATION RESULTS")
    print(f"Test Set: Uniprot_Test_Set.csv ({len(test_sequences)} sequences)")
    print("="*100)
    print()
    
    # Format table
    table = df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'F1-Macro', 'MCC', 'AUC']:
        table[col] = table[col].apply(lambda x: f"{x:.4f}")
    
    print(table.to_string(index=False))
    print()
    
    # Find best model
    best_idx = df['F1'].idxmax()
    best_result = df.loc[best_idx]
    
    print("="*100)
    print("BEST MODEL ON TEST SET:")
    print(f"  Encoding: {best_result['Encoding']}")
    print(f"  Model: {best_result['Model']}")
    print(f"  F1 Score: {best_result['F1']:.4f}")
    print(f"  Accuracy: {best_result['Accuracy']:.4f}")
    print(f"  MCC: {best_result['MCC']:.4f}")
    print(f"  AUC: {best_result['AUC']:.4f}")
    print("="*100)
    
    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    csv_file = output_dir / 'test_set_evaluation.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"\n✓ Results saved to: {csv_file}")
    
    txt_file = output_dir / 'test_set_evaluation.txt'
    with open(txt_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write(f"Test Set: Uniprot_Test_Set.csv ({len(test_sequences)} sequences)\n")
        f.write("="*100 + "\n\n")
        f.write(table.to_string(index=False) + "\n\n")
        f.write("="*100 + "\n")
        f.write("BEST MODEL ON TEST SET:\n")
        f.write(f"  Encoding: {best_result['Encoding']}\n")
        f.write(f"  Model: {best_result['Model']}\n")
        f.write(f"  F1 Score: {best_result['F1']:.4f}\n")
        f.write(f"  Accuracy: {best_result['Accuracy']:.4f}\n")
        f.write(f"  MCC: {best_result['MCC']:.4f}\n")
        f.write(f"  AUC: {best_result['AUC']:.4f}\n")
        f.write("="*100 + "\n")
    
    print(f"✓ Summary saved to: {txt_file}")
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
