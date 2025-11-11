"""
Evaluation script for RCL prediction models.
"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import (
    read_csv_with_annotations,
    get_encoder,
    EncodedDataset
)
from models import get_model
from utils import (
    compute_metrics,
    compute_confusion_matrix,
    compute_per_protein_errors,
    plot_confusion_matrix,
    plot_error_histogram,
    save_error_histogram_csv
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RCL prediction model')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to model directory (e.g., runs/run_001)')
    parser.add_argument('--test-file', type=str,
                       help='Path to test CSV file (overrides config)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device (overrides config)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: model_dir/evaluation)')
    
    return parser.parse_args()


def load_model(model_dir: Path, device: str):
    """Load trained model from checkpoint."""
    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Get encoding dimension
    if config['encoding'] == 'onehot':
        encoding_dim = 21
    elif config['encoding'] == 'blosum':
        encoding_dim = 20
    elif config['encoding'] == 'esm2':
        encoding_dim = 1280
    else:
        raise ValueError(f"Unknown encoding: {config['encoding']}")
    
    # Create model
    model = get_model(
        config['model'],
        encoding_dim,
        config['models'][config['model']],
        num_classes=2
    )
    
    # Load weights
    checkpoint = torch.load(model_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(model, data_loader, device):
    """Run model evaluation."""
    all_preds = []
    all_labels = []
    all_ids = []
    all_sequences = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            encodings = batch['encoding'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(encodings)
            probs = torch.softmax(outputs, dim=-1)
            
            # Collect results
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_ids.extend(batch['id'])
            if 'sequence' in batch:
                all_sequences.extend(batch['sequence'])
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_preds, all_labels, all_ids, all_sequences


def extract_rcl_predictions(pred_probs: np.ndarray, threshold: float = 0.5):
    """Extract RCL start/end positions from predictions."""
    # Get RCL predictions (class 1)
    rcl_pred = (pred_probs[:, 1] > threshold).astype(int)
    
    # Find consecutive regions
    rcl_indices = np.where(rcl_pred == 1)[0]
    
    if len(rcl_indices) == 0:
        return None, None, 0.0
    
    # Find start and end of longest consecutive region
    breaks = np.where(np.diff(rcl_indices) > 1)[0]
    
    if len(breaks) == 0:
        # Single continuous region
        rcl_start = int(rcl_indices[0])
        rcl_end = int(rcl_indices[-1]) + 1
        confidence = float(pred_probs[rcl_indices, 1].mean())
    else:
        # Multiple regions, take the longest
        regions = []
        start_idx = 0
        for break_idx in breaks:
            end_idx = break_idx + 1
            region = rcl_indices[start_idx:end_idx]
            regions.append(region)
            start_idx = end_idx
        regions.append(rcl_indices[start_idx:])
        
        # Get longest region
        longest_region = max(regions, key=len)
        rcl_start = int(longest_region[0])
        rcl_end = int(longest_region[-1]) + 1
        confidence = float(pred_probs[longest_region, 1].mean())
    
    return rcl_start, rcl_end, confidence


def main():
    args = parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model and config
    print(f"Loading model from {model_dir}...")
    model, config = load_model(model_dir, device)
    
    # Determine test file
    if args.test_file:
        test_file = args.test_file
    else:
        test_file = config['data']['test_serpin']
    
    print(f"Test file: {test_file}")
    
    # Load test data
    print("Loading test data...")
    test_ids, test_labels, test_seqs, _ = read_csv_with_annotations(
        test_file,
        max_length=config['data']['max_length']
    )
    
    print(f"Test sequences: {len(test_seqs)}")
    
    # Encode sequences
    print(f"Encoding sequences with {config['encoding']}...")
    encoder = get_encoder(
        config['encoding'],
        max_length=config['data']['max_length'],
        device=device
    )
    
    if config['encoding'] == 'esm2':
        test_encodings = encoder.encode_batch(
            test_seqs,
            batch_size=config['encoding']['esm2']['batch_size']
        )
    else:
        test_encodings = encoder.encode_batch(test_seqs)
    
    # Create dataset and loader
    test_dataset = EncodedDataset(
        test_encodings,
        test_labels,
        test_ids,
        test_seqs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    print("\nEvaluating model...")
    pred_probs, true_labels, ids, sequences = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(true_labels, pred_probs)
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")
    print("="*50)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_dir / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Confusion matrix
    cm = compute_confusion_matrix(true_labels, pred_probs)
    plot_confusion_matrix(cm, str(output_dir / 'confusion_matrix.png'))
    
    # Error histogram
    errors = compute_per_protein_errors(true_labels, pred_probs)
    plot_error_histogram(errors, str(output_dir / 'error_histogram.png'))
    save_error_histogram_csv(errors, str(output_dir / 'error_counts.csv'))
    
    # Save detailed predictions
    print("\nSaving detailed predictions...")
    results = []
    
    for i, (protein_id, sequence) in enumerate(zip(ids, sequences)):
        # Get predictions for this protein
        pred = pred_probs[i]
        true = true_labels[i]
        
        # Extract predicted RCL
        rcl_start, rcl_end, confidence = extract_rcl_predictions(pred[:len(sequence)])
        
        # Get true RCL
        true_rcl_mask = true[:len(sequence), 1] == 1
        true_rcl_indices = np.where(true_rcl_mask)[0]
        
        if len(true_rcl_indices) > 0:
            true_rcl_start = int(true_rcl_indices[0]) + 1  # Convert to 1-indexed
            true_rcl_end = int(true_rcl_indices[-1]) + 1
        else:
            true_rcl_start = None
            true_rcl_end = None
        
        # Convert to 1-indexed for output
        if rcl_start is not None:
            rcl_start += 1
        
        # Count errors
        true_classes = np.argmax(true[:len(sequence)], axis=-1)
        pred_classes = np.argmax(pred[:len(sequence)], axis=-1)
        num_errors = int(np.sum(true_classes != pred_classes))
        
        results.append({
            'protein_id': protein_id,
            'sequence': sequence,
            'seq_length': len(sequence),
            'true_rcl_start': true_rcl_start,
            'true_rcl_end': true_rcl_end,
            'pred_rcl_start': rcl_start,
            'pred_rcl_end': rcl_end,
            'confidence': confidence if confidence else 0.0,
            'num_errors': num_errors
        })
    
    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Total proteins: {len(results)}")
    print(f"Proteins with predicted RCL: {sum(1 for r in results if r['pred_rcl_start'] is not None)}")
    print(f"Mean confidence: {np.mean([r['confidence'] for r in results]):.4f}")
    print(f"Mean errors per protein: {np.mean([r['num_errors'] for r in results]):.2f}")
    print("="*50)
    
    print(f"\n✓ Evaluation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
