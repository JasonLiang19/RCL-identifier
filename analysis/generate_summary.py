#!/usr/bin/env python3
"""
Generate summary table from experiment results.
Usage: python analysis/generate_summary.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def extract_best_metrics(run_dir):
    """Extract metrics from the epoch with best F1 score."""
    history_file = Path(run_dir) / 'training_history.json'
    
    if not history_file.exists():
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Find epoch with best F1
    val_metrics = history['val_metrics']
    f1_scores = val_metrics['f1']
    best_epoch = np.argmax(f1_scores)
    
    # Extract all metrics at best epoch
    metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'mcc', 'auc']:
        if metric in val_metrics:
            metrics[metric] = val_metrics[metric][best_epoch]
        else:
            metrics[metric] = 0.0
    
    metrics['best_epoch'] = best_epoch + 1
    metrics['total_epochs'] = len(f1_scores)
    
    return metrics


def generate_summary_table(results_dir='analysis/results'):
    """Generate summary table from all experiment results."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return None
    
    # Find all result directories
    result_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    if not result_dirs:
        print(f"Error: No result directories found in {results_dir}")
        return None
    
    # Expected experiments
    expected_experiments = [
        ('onehot', 'cnn'),
        ('onehot', 'unet'),
        ('blosum', 'cnn'),
        ('blosum', 'unet'),
        ('esm2_650m', 'cnn'),
        ('esm2_650m', 'unet'),
    ]
    
    # Track which experiments are found
    found_experiments = set()
    missing_experiments = []
    incomplete_runs = []
    
    # Collect data
    data = []
    for result_dir in sorted(result_dirs):
        # Parse encoding and model from directory name
        parts = result_dir.name.split('_')
        if len(parts) < 2:
            continue
        
        # Handle esm2_650m case
        if 'esm2' in parts[0]:
            encoding = '_'.join(parts[:-1])  # esm2_650m
            model = parts[-1]
        else:
            encoding = parts[0]
            model = parts[1]
        
        metrics = extract_best_metrics(result_dir)
        
        if metrics is None:
            incomplete_runs.append(f"{encoding}_{model}")
            continue
        
        found_experiments.add((encoding, model))
        
        row = {
            'Encoding': encoding,
            'Model': model.upper(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'F1-Macro': metrics['f1_macro'],
            'MCC': metrics['mcc'],
            'AUC': metrics['auc'],
            'Best Epoch': metrics['best_epoch'],
            'Total Epochs': metrics['total_epochs']
        }
        data.append(row)
    
    # Determine missing experiments
    for enc, mod in expected_experiments:
        if (enc, mod) not in found_experiments:
            missing_experiments.append(f"{enc}_{mod}")
    
    # Print status report
    print(f"Found {len(data)} completed experiments out of {len(expected_experiments)} expected")
    if incomplete_runs:
        print(f"⚠ Incomplete runs (skipped): {', '.join(incomplete_runs)}")
    if missing_experiments:
        print(f"⚠ Missing experiments: {', '.join(missing_experiments)}")
    print()
    
    if not data:
        print("Error: No valid results found")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by encoding and model
    encoding_order = {'onehot': 0, 'blosum': 1, 'esm2_650m': 2}
    df['enc_order'] = df['Encoding'].map(encoding_order)
    df = df.sort_values(['enc_order', 'Model']).drop('enc_order', axis=1)
    
    return df


def print_summary_table(df, output_file=None):
    """Print formatted summary table."""
    
    num_results = len(df)
    
    header = f"""
========================================
RCL Prediction Results Summary
({num_results} experiment{'s' if num_results != 1 else ''} completed)
========================================
"""
    
    print(header)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(header + '\n')
    
    # Format table
    table = df.copy()
    
    # Format numeric columns to 4 decimal places
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'F1-Macro', 'MCC', 'AUC']:
        table[col] = table[col].apply(lambda x: f"{x:.4f}")
    
    # Print main metrics table
    main_cols = ['Encoding', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'F1-Macro', 'MCC', 'AUC']
    main_table = table[main_cols].to_string(index=False)
    
    print(main_table)
    print()
    
    if output_file:
        with open(output_file, 'a') as f:
            f.write(main_table + '\n\n')
    
    # Print training info
    training_info = table[['Encoding', 'Model', 'Best Epoch', 'Total Epochs']].to_string(index=False)
    print("Training Information:")
    print(training_info)
    print()
    
    if output_file:
        with open(output_file, 'a') as f:
            f.write("Training Information:\n")
            f.write(training_info + '\n\n')
    
    # Find best overall (only if we have results)
    if num_results > 0:
        best_idx = df['F1'].idxmax()
        best_result = df.loc[best_idx]
        
        best_summary = f"""
Best Performance (among completed experiments):
  Encoding: {best_result['Encoding']}
  Model: {best_result['Model']}
  F1 Score: {best_result['F1']:.4f}
  Accuracy: {best_result['Accuracy']:.4f}
  MCC: {best_result['MCC']:.4f}
"""
        
        print(best_summary)
        
        if output_file:
            with open(output_file, 'a') as f:
                f.write(best_summary + '\n')
        print(f"✓ Summary saved to: {output_file}")
    
    footer = """
========================================
"""
    print(footer)


def export_csv(df, output_file='analysis/results/summary.csv'):
    """Export summary to CSV."""
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"✓ CSV exported to: {output_file}")


def main():
    results_dir = 'analysis/results'
    
    print("Generating summary from experiment results...")
    print()
    
    df = generate_summary_table(results_dir)
    
    if df is None:
        print("Failed to generate summary table")
        sys.exit(1)
    
    # Print summary
    print_summary_table(df, output_file='analysis/results/summary.txt')
    
    # Export CSV
    export_csv(df, output_file='analysis/results/summary.csv')
    
    print()
    if len(df) < 6:
        print(f"Note: Summary includes {len(df)}/6 experiments. Run remaining experiments to complete the comparison.")
    print("Done!")



if __name__ == '__main__':
    main()
