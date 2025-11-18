#!/usr/bin/env python3
"""
Combined evaluation script for RCL prediction models.

Supports both sequence-level (default) and residue-level (--aa flag) evaluation.

Sequence-level metrics:
    - Binary classification: Does sequence have RCL? (Accuracy, Precision, Recall, F1)
    - IoU (Intersection over Union): Overlap between predicted and true RCL regions
    - Exact match: Percentage of sequences with exact RCL position prediction

Residue-level metrics (--aa flag):
    - Per-residue classification metrics across all positions
    - Same as original evaluate_test_set.py

Usage:
    # Sequence-level evaluation on test set (default)
    python evaluate_combined.py --dataset test
    
    # Residue-level evaluation on test set
    python evaluate_combined.py --dataset test --aa
    
    # Sequence-level on validation set
    python evaluate_combined.py --dataset val
    
    # Single model evaluation
    python evaluate_combined.py --dataset test --model esm2_650m_unet
    
    # With minimum consecutive filter
    python evaluate_combined.py --dataset test --min-consecutive 12
"""

import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.encoders import BLOSUMEncoder, ESM2_650M_Encoder, EncodedDataset, OneHotEncoder
from src.models.architectures import CNNModel, UNetModel, LSTMModel

def get_latest_run_dir(base_dir: str) -> Path:
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"No run directory found at: {base_dir}")

    run_dirs = [
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]

    if not run_dirs:
        raise FileNotFoundError("No run directories found")

    # Sort run directories numerically: run_001 < run_002 < run_010 < ...
    run_dirs_sorted = sorted(
        run_dirs,
        key=lambda d: int(d.name.split("_")[1])
    )

    return run_dirs_sorted[-1]   # Most recent run directory

def load_model_and_encoder(model_name, device):
    """Load trained model and corresponding encoder."""
    results_dir = Path(__file__).parent / 'results' / model_name
    
    # Load config
    with open(results_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Determine encoding from model name
    if 'onehot' in model_name:
        encoding = 'onehot'
    elif 'blosum' in model_name:
        encoding = 'blosum'
    elif 'esm2' in model_name:
        encoding = 'esm2_650m'
    else:
        raise ValueError(f"Cannot determine encoding from model name: {model_name}")
    
    # Determine model type from model name or config
    if 'unet' in model_name:
        model_type = 'UNET'
    elif 'cnn' in model_name:
        model_type = 'CNN'
    elif 'lstm' in model_name:
        model_type = 'LSTM'
    else:
        model_type = config.get('model', 'CNN').upper()
    
    # Initialize encoder
    if encoding == 'onehot':
        encoder = OneHotEncoder(max_length=1024)
    elif encoding == 'blosum':
        encoder = BLOSUMEncoder(max_length=1024)
    elif encoding == 'esm2_650m':
        encoder = ESM2_650M_Encoder(max_length=1024)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    
    # Initialize model
    if model_type == 'CNN':
        model_class = CNNModel
    elif model_type == 'UNET':
        model_class = UNetModel
    elif model_type == 'LSTM':
        model_class = LSTMModel
    else:
        raise ValueError(f"Unknown MODEL")
    
    # Get embedding dimension from encoder
    if hasattr(encoder, 'embedding_dim'):
        input_dim = encoder.embedding_dim
    elif hasattr(encoder, 'dim'):
        input_dim = encoder.dim
    else:
        input_dim = encoder.get_encoding_dim()
    
    model = model_class(
        input_dim=input_dim,
        num_classes=2
    )
    
    # Load weights
    results_dir = get_latest_run_dir(results_dir)
    model_path = results_dir / 'best_model.pt'
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Create simple config dict for return
    simple_config = {
        'encoding': encoding,
        'model_type': model_type
    }
    
    return model, encoder, simple_config


def load_data(dataset_type, data_dir):
    """
    Load dataset for evaluation.
    
    Args:
        dataset_type: 'test' or 'val'
        data_dir: Path to data directory
        
    Returns:
        List of (id, label, sequence, rcl_sequence) tuples
    """
    from src.data.data_loader import read_csv_with_annotations
    
    if dataset_type == 'test':
        csv_file = data_dir / 'raw' / 'Uniprot_Test_Set.csv'
        print(f"\nLoading test set from {csv_file}")
        ids, labels, sequences, rcl_sequences = read_csv_with_annotations(csv_file)
    elif dataset_type == 'val':
        # Load validation split from training data
        csv_file = data_dir / 'raw' / 'Alphafold_RCL_annotations.csv'
        print(f"\nLoading validation set from {csv_file}")
        
        # Read all data
        ids, labels, sequences, rcl_sequences = read_csv_with_annotations(csv_file)
        
        # Use same random seed as training to get consistent validation split
        np.random.seed(42)
        total_samples = len(ids)
        indices = np.random.permutation(total_samples)
        val_size = int(0.2 * total_samples)
        val_indices = indices[-val_size:]
        
        # Extract validation subset
        ids = [ids[i] for i in val_indices]
        labels = [labels[i] for i in val_indices]
        sequences = [sequences[i] for i in val_indices]
        rcl_sequences = [rcl_sequences[i] for i in val_indices]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Loaded {len(ids)} sequences")
    
    return list(zip(ids, labels, sequences, rcl_sequences))


def extract_rcl_region(pred_probs, seq_length, threshold=0.5, min_consecutive=1):
    """
    Extract RCL start/end positions from predictions.
    Same as predict.py implementation.
    """
    # Get RCL predictions (class 1) for actual sequence length
    rcl_pred = (pred_probs[:seq_length, 1] > threshold).astype(int)
    
    # Find indices where RCL is predicted
    rcl_indices = np.where(rcl_pred == 1)[0]
    
    if len(rcl_indices) == 0:
        return None, None, 0, 0.0
    
    # Find consecutive regions
    if len(rcl_indices) == 1:
        # Single residue - check minimum length
        if min_consecutive > 1:
            return None, None, 0, 0.0
        return int(rcl_indices[0]), int(rcl_indices[0]) + 1, 1, float(pred_probs[rcl_indices[0], 1])
    
    # Find breaks in consecutive indices
    breaks = np.where(np.diff(rcl_indices) > 1)[0]
    
    if len(breaks) == 0:
        # Single continuous region
        rcl_start = int(rcl_indices[0])
        rcl_end = int(rcl_indices[-1]) + 1
        rcl_length = rcl_end - rcl_start
        confidence = float(pred_probs[rcl_indices, 1].mean())
        
        # Check minimum length
        if rcl_length < min_consecutive:
            return None, None, 0, 0.0
    else:
        # Multiple regions - filter by minimum length, then take the longest one
        regions = []
        
        start_idx = 0
        for break_idx in breaks:
            end_idx = break_idx + 1
            region = rcl_indices[start_idx:end_idx]
            if len(region) >= min_consecutive:  # Filter by minimum length
                regions.append(region)
            start_idx = end_idx
        
        # Last region
        last_region = rcl_indices[start_idx:]
        if len(last_region) >= min_consecutive:
            regions.append(last_region)
        
        # No valid regions found
        if len(regions) == 0:
            return None, None, 0, 0.0
        
        # Find longest region
        longest_region = max(regions, key=len)
        
        rcl_start = int(longest_region[0])
        rcl_end = int(longest_region[-1]) + 1
        rcl_length = rcl_end - rcl_start
        confidence = float(pred_probs[longest_region, 1].mean())
    
    return rcl_start, rcl_end, rcl_length, confidence


def calculate_iou(pred_start, pred_end, true_start, true_end):
    """Calculate Intersection over Union between predicted and true RCL regions."""
    if pred_start is None or true_start is None:
        return 0.0
    
    # Calculate intersection
    inter_start = max(pred_start, true_start)
    inter_end = min(pred_end, true_end)
    intersection = max(0, inter_end - inter_start)
    
    # Calculate union
    union_start = min(pred_start, true_start)
    union_end = max(pred_end, true_end)
    union = union_end - union_start
    
    return intersection / union if union > 0 else 0.0


def evaluate_sequence_level(model, encoder, data, device, batch_size=8, threshold=0.5, min_consecutive=1):
    """
    Evaluate model using sequence-level metrics.
    
    Metrics:
        - Binary classification: Accuracy, Precision, Recall, F1 for "has RCL" prediction
        - IoU: Mean Intersection over Union for predicted vs true RCL positions
        - Exact match: Percentage with exact position prediction
    """
    print("\n" + "="*80)
    print("SEQUENCE-LEVEL EVALUATION")
    print("="*80)
    
    # Extract sequences and encode
    ids = [d[0] for d in data]
    sequences = [d[2] for d in data]
    rcl_sequences = [d[3] for d in data]
    
    print(f"\nEncoding {len(sequences)} sequences...")
    
    if isinstance(encoder, ESM2_650M_Encoder):
        encodings = encoder.encode_batch(sequences, batch_size=1)
    else:
        encodings = encoder.encode_batch(sequences)
    
    # Create dataset and loader
    labels = np.zeros((len(encodings), encoder.max_length, 2))  # Placeholder
    dataset = EncodedDataset(encodings, labels, ids=ids, sequences=sequences, lazy_load=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Make predictions
    print("\nMaking predictions...")
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            encodings_batch = batch['encoding'].to(device)
            outputs = model(encodings_batch)
            probs = torch.softmax(outputs, dim=-1)
            all_preds.append(probs.cpu().numpy())
    
    pred_probs = np.concatenate(all_preds, axis=0)
    
    # Extract RCL regions and calculate metrics
    print("\nCalculating sequence-level metrics...")
    
    binary_true = []  # Does sequence have RCL?
    binary_pred = []
    ious = []
    exact_matches = 0
    total_sequences = len(data)
    
    for i, (seq_id, label, sequence, rcl_seq) in enumerate(tqdm(data, desc='Analyzing')):
        # True RCL position
        true_has_rcl = (rcl_seq != '' and rcl_seq != 'N/A')
        binary_true.append(1 if true_has_rcl else 0)
        
        if true_has_rcl:
            # Find true RCL position in sequence
            rcl_start_true = sequence.find(rcl_seq)
            if rcl_start_true == -1:
                print(f"Warning: RCL sequence not found in {seq_id}")
                rcl_start_true = 0
                rcl_end_true = 0
            else:
                rcl_end_true = rcl_start_true + len(rcl_seq)
        else:
            rcl_start_true = None
            rcl_end_true = None
        
        # Predicted RCL position
        rcl_start_pred, rcl_end_pred, rcl_length, confidence = extract_rcl_region(
            pred_probs[i],
            len(sequence),
            threshold=threshold,
            min_consecutive=min_consecutive
        )
        
        pred_has_rcl = (rcl_start_pred is not None)
        binary_pred.append(1 if pred_has_rcl else 0)
        
        # Calculate IoU if both have RCL
        if true_has_rcl and pred_has_rcl:
            iou = calculate_iou(rcl_start_pred, rcl_end_pred, rcl_start_true, rcl_end_true)
            ious.append(iou)
            
            # Check exact match (within 2 residues tolerance)
            if abs(rcl_start_pred - rcl_start_true) <= 2 and abs(rcl_end_pred - rcl_end_true) <= 2:
                exact_matches += 1
        elif true_has_rcl and not pred_has_rcl:
            ious.append(0.0)  # Missed RCL
        # If no true RCL, we don't include in IoU calculation
    
    # Binary classification metrics
    binary_true = np.array(binary_true)
    binary_pred = np.array(binary_pred)
    
    accuracy = accuracy_score(binary_true, binary_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_true, binary_pred, average='binary', zero_division=0
    )
    
    # IoU and exact match
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
    exact_match_pct = (exact_matches / total_sequences) * 100
    
    # Count statistics
    num_true_rcl = int(binary_true.sum())
    num_pred_rcl = int(binary_pred.sum())
    true_positives = int(((binary_true == 1) & (binary_pred == 1)).sum())
    false_positives = int(((binary_true == 0) & (binary_pred == 1)).sum())
    false_negatives = int(((binary_true == 1) & (binary_pred == 0)).sum())
    
    results = {
        'binary_accuracy': accuracy,
        'binary_precision': precision,
        'binary_recall': recall,
        'binary_f1': f1,
        'mean_iou': mean_iou,
        'exact_match_pct': exact_match_pct,
        'num_sequences': total_sequences,
        'num_true_rcl': num_true_rcl,
        'num_pred_rcl': num_pred_rcl,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
    
    return results


def evaluate_residue_level(model, encoder, data, device, batch_size=8):
    """
    Evaluate model using residue-level metrics (original per-position evaluation).
    """
    print("\n" + "="*80)
    print("RESIDUE-LEVEL EVALUATION")
    print("="*80)
    
    # Extract sequences and encode
    ids = [d[0] for d in data]
    sequences = [d[2] for d in data]
    
    print(f"\nEncoding {len(sequences)} sequences...")
    
    if isinstance(encoder, ESM2_650M_Encoder):
        encodings = encoder.encode_batch(sequences, batch_size=1)
    else:
        encodings = encoder.encode_batch(sequences)
    
    # Create labels (same as training)
    labels = []
    for seq_id, label, sequence, rcl_seq in data:
        # Create binary labels: [non-RCL, RCL]
        label_array = np.zeros((encoder.max_length, 2), dtype=np.float32)
        label_array[:, 0] = 1  # Initialize as non-RCL
        
        # Mark RCL region if valid annotation exists
        if rcl_seq and rcl_seq != '' and rcl_seq != 'N/A':
            rcl_start = sequence.find(rcl_seq)
            if rcl_start != -1:
                rcl_end = rcl_start + len(rcl_seq)
                for i in range(rcl_start, min(rcl_end, len(sequence))):
                    label_array[i] = [0, 1]
        
        # Mask positions beyond sequence length
        for i in range(len(sequence), encoder.max_length):
            label_array[i] = [9999, 9999]  # Masked value
        
        labels.append(label_array)
    
    labels = np.array(labels)
    
    # Create dataset and loader
    dataset = EncodedDataset(encodings, labels, ids=ids, sequences=sequences, lazy_load=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Make predictions and collect labels
    print("\nMaking predictions...")
    all_preds = []
    all_labels = []
    all_seq_lengths = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            encodings_batch = batch['encoding'].to(device)
            labels_batch = batch['label']
            
            outputs = model(encodings_batch)
            probs = torch.softmax(outputs, dim=-1)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels_batch.numpy())
            
            # Track actual sequence lengths
            for seq in batch['sequence']:
                all_seq_lengths.append(len(seq))
    
    pred_probs = np.concatenate(all_preds, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    
    # Flatten predictions and labels (only for actual sequence positions)
    print("\nCalculating residue-level metrics...")
    
    all_pred_flat = []
    all_true_flat = []
    all_prob_flat = []
    
    for i in range(len(pred_probs)):
        seq_len = all_seq_lengths[i]
        
        # Get predictions and labels for actual sequence positions
        pred_class = np.argmax(pred_probs[i, :seq_len], axis=-1)
        true_class = np.argmax(true_labels[i, :seq_len], axis=-1)
        prob_rcl = pred_probs[i, :seq_len, 1]  # Probability of RCL class
        
        all_pred_flat.extend(pred_class)
        all_true_flat.extend(true_class)
        all_prob_flat.extend(prob_rcl)
    
    all_pred_flat = np.array(all_pred_flat)
    all_true_flat = np.array(all_true_flat)
    all_prob_flat = np.array(all_prob_flat)
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_flat, all_pred_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_flat, all_pred_flat, average='binary', zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        all_true_flat, all_pred_flat, average='macro', zero_division=0
    )
    mcc = matthews_corrcoef(all_true_flat, all_pred_flat)
    auc = roc_auc_score(all_true_flat, all_prob_flat)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1_macro,
        'mcc': mcc,
        'auc': auc
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate RCL prediction models with sequence-level or residue-level metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--dataset', type=str, default='test', choices=['test', 'val'],
                       help='Dataset to evaluate on (default: test)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to evaluate (default: all 6 models)')
    parser.add_argument('--aa', action='store_true',
                       help='Use residue-level evaluation (default: sequence-level)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for prediction (default: 0.5)')
    parser.add_argument('--min-consecutive', type=int, default=1,
                       help='Minimum consecutive residues for valid RCL (default: 1, use 12 for filtering)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference (default: 8)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent / 'results'
    
    # Load data
    data = load_data(args.dataset, data_dir)
    
    # Determine which models to evaluate
    if args.model:
        models_to_eval = [args.model]
    else:
        models_to_eval = [
            'onehot_cnn',
            'onehot_unet',
            'onehot_lstm',
            'blosum_cnn',
            'blosum_unet',
            'blosum_lstm',
            'esm2_650m_cnn',
            'esm2_650m_unet',
            'esm2_650m_lstm'
        ]
    
    # Evaluate each model
    all_results = []
    
    for model_name in models_to_eval:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")
        
        # Load model
        model, encoder, config = load_model_and_encoder(model_name, device)
        
        # Evaluate
        if args.aa:
            # Residue-level evaluation
            results = evaluate_residue_level(model, encoder, data, device, args.batch_size)
            results['encoding'] = config['encoding']
            results['model'] = config['model_type']
        else:
            # Sequence-level evaluation
            results = evaluate_sequence_level(
                model, encoder, data, device, args.batch_size, args.threshold, args.min_consecutive
            )
            results['encoding'] = config['encoding']
            results['model'] = config['model_type']
        
        all_results.append(results)
        
        # Print results
        print("\nResults:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Save and display comparison table
    print("\n" + "="*80)
    print(f"{args.dataset.upper()} SET EVALUATION RESULTS")
    print(f"Evaluation Mode: {'Residue-level (--aa)' if args.aa else 'Sequence-level'}")
    if not args.aa:
        print(f"Threshold: {args.threshold}, Min Consecutive: {args.min_consecutive}")
    print("="*80)
    print()
    
    df = pd.DataFrame(all_results)
    
    if args.aa:
        # Residue-level table
        display_df = df[['encoding', 'model', 'accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'mcc', 'auc']].copy()
        display_df.columns = ['Encoding', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'F1-Macro', 'MCC', 'AUC']
        
        # Find best by sorting by f1 first, then breaking ties with MCC
        best = df.sort_values(
            by=['f1', 'mcc'],
            ascending=[False, False]
        ).iloc[0]
    else:
        # Sequence-level table
        display_df = df[['encoding', 'model', 'binary_accuracy', 'binary_precision', 'binary_recall', 
                        'binary_f1', 'mean_iou', 'exact_match_pct']].copy()
        display_df.columns = ['Encoding', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Mean IoU', 'Exact Match %']
        
        # Find best by sorting by exact match first, then breaking ties with f1
        best = df.sort_values(
            by=['exact_match_pct', 'binary_f1'],
            ascending=[False, False]
        ).iloc[0]
    
    # Format and display
    for col in display_df.columns[2:]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    print()
    
    if args.aa:
        print(f"BEST MODEL: {best['encoding']} + {best['model']} (F1 = {best['f1']:.4f}, MCC = {best['mcc']:.4f})")
    else:
        print(f"BEST MODEL: {best['encoding']} + {best['model']} (Exact Match % = {best['exact_match_pct']:.4f}, F1 = {best['binary_f1']:.4f})")
    print()
    
    # include min_consecutive in filename
    min_consecutive = ""
    if args.min_consecutive != 1:
        min_consecutive = f"min{args.min_consecutive}_"

    # Save results
    eval_type = 'residue_level' if args.aa else 'sequence_level'
    output_file = results_dir / f'{args.dataset}_{eval_type}_{min_consecutive}evaluation.csv'
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"✓ Results saved to: {output_file}")
    
    # Save formatted summary
    summary_file = results_dir / f'{args.dataset}_{eval_type}_{min_consecutive}evaluation.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{args.dataset.upper()} SET EVALUATION RESULTS\n")
        f.write(f"Evaluation Mode: {'Residue-level (--aa)' if args.aa else 'Sequence-level'}\n")
        if not args.aa:
            f.write(f"Threshold: {args.threshold}, Min Consecutive: {args.min_consecutive}\n")
        f.write("="*80 + "\n\n")
        f.write(display_df.to_string(index=False) + "\n\n")
        
        if args.aa:
            f.write(f"BEST MODEL: {best['encoding']} + {best['model']} (F1 = {best['f1']:.4f}, MCC = {best['mcc']:.4f})\n")
        else:
            f.write(f"BEST MODEL: {best['encoding']} + {best['model']} (Exact Match % = {best['exact_match_pct']:.4f}, F1 = {best['binary_f1']:.4f})\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
