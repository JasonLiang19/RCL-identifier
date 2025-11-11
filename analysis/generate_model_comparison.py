#!/usr/bin/env python3
"""
Generate comprehensive comparison table of all 6 models.
Shows both validation set (from training) and test set performance.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_validation_metrics():
    """Load validation metrics from training history."""
    results_dir = Path(__file__).parent / 'results'
    
    models = [
        'onehot_cnn',
        'onehot_unet',
        'blosum_cnn',
        'blosum_unet',
        'esm2_650m_cnn',
        'esm2_650m_unet'
    ]
    
    val_results = []
    
    for model_name in models:
        history_file = results_dir / model_name / 'training_history.json'
        
        if not history_file.exists():
            print(f"Warning: {history_file} not found")
            continue
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Find epoch with best F1
        val_metrics = history['val_metrics']
        f1_scores = val_metrics['f1']
        best_epoch = np.argmax(f1_scores)
        
        # Extract metrics at best epoch
        encoding_name = model_name.rsplit('_', 1)[0]
        model_type = model_name.rsplit('_', 1)[1]
        
        val_results.append({
            'Encoding': encoding_name,
            'Model': model_type.upper(),
            'Val_Accuracy': val_metrics['accuracy'][best_epoch],
            'Val_Precision': val_metrics['precision'][best_epoch],
            'Val_Recall': val_metrics['recall'][best_epoch],
            'Val_F1': val_metrics['f1'][best_epoch],
            'Val_F1_Macro': val_metrics['f1_macro'][best_epoch],
            'Val_MCC': val_metrics['mcc'][best_epoch],
            'Val_AUC': val_metrics['auc'][best_epoch],
            'Best_Epoch': best_epoch + 1,
            'Total_Epochs': len(f1_scores)
        })
    
    return pd.DataFrame(val_results)


def load_test_metrics():
    """Load test set metrics."""
    test_file = Path(__file__).parent / 'results' / 'test_set_evaluation.csv'
    
    if not test_file.exists():
        print(f"Warning: Test results not found at {test_file}")
        return None
    
    df = pd.read_csv(test_file)
    
    # Rename columns for clarity
    df = df.rename(columns={
        'Accuracy': 'Test_Accuracy',
        'Precision': 'Test_Precision',
        'Recall': 'Test_Recall',
        'F1': 'Test_F1',
        'F1-Macro': 'Test_F1_Macro',
        'MCC': 'Test_MCC',
        'AUC': 'Test_AUC'
    })
    
    return df


def main():
    print("Generating model comparison table...")
    print()
    
    # Load validation metrics
    val_df = load_validation_metrics()
    
    # Load test metrics
    test_df = load_test_metrics()
    
    # Merge if test results exist
    if test_df is not None:
        df = pd.merge(val_df, test_df, on=['Encoding', 'Model'])
    else:
        df = val_df
        print("Note: Test results not available. Showing validation metrics only.")
        print()
    
    # Sort by encoding and model
    encoding_order = {'onehot': 0, 'blosum': 1, 'esm2_650m': 2}
    df['enc_order'] = df['Encoding'].map(encoding_order)
    df = df.sort_values(['enc_order', 'Model']).drop('enc_order', axis=1)
    
    # Save full comparison
    output_file = Path(__file__).parent / 'results' / 'model_comparison.csv'
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"✓ Full comparison saved to: {output_file}")
    
    # Create formatted tables
    print("\n" + "="*120)
    print("MODEL COMPARISON: VALIDATION SET PERFORMANCE")
    print(f"Dataset: 3,431 sequences (80/20 train/val split)")
    print("="*120)
    print()
    
    # Validation table
    val_table = df[['Encoding', 'Model', 'Val_Accuracy', 'Val_Precision', 'Val_Recall', 
                    'Val_F1', 'Val_F1_Macro', 'Val_MCC', 'Val_AUC']].copy()
    
    # Format numbers
    for col in val_table.columns[2:]:
        val_table[col] = val_table[col].apply(lambda x: f"{x:.4f}")
    
    print(val_table.to_string(index=False))
    print()
    
    # Find best on validation
    best_val_idx = df['Val_F1'].idxmax()
    best_val = df.loc[best_val_idx]
    print(f"Best Validation F1: {best_val['Encoding']} + {best_val['Model']} = {best_val['Val_F1']:.4f}")
    print()
    
    # Test table if available
    if test_df is not None:
        print("="*120)
        print("MODEL COMPARISON: TEST SET PERFORMANCE")
        print(f"Dataset: Uniprot_Test_Set.csv (78 sequences)")
        print("="*120)
        print()
        
        test_table = df[['Encoding', 'Model', 'Test_Accuracy', 'Test_Precision', 'Test_Recall',
                        'Test_F1', 'Test_F1_Macro', 'Test_MCC', 'Test_AUC']].copy()
        
        # Format numbers
        for col in test_table.columns[2:]:
            test_table[col] = test_table[col].apply(lambda x: f"{x:.4f}")
        
        print(test_table.to_string(index=False))
        print()
        
        # Find best on test
        best_test_idx = df['Test_F1'].idxmax()
        best_test = df.loc[best_test_idx]
        print(f"Best Test F1: {best_test['Encoding']} + {best_test['Model']} = {best_test['Test_F1']:.4f}")
        print()
    
    # Training info
    print("="*120)
    print("TRAINING INFORMATION")
    print("="*120)
    print()
    
    training_info = df[['Encoding', 'Model', 'Best_Epoch', 'Total_Epochs']].copy()
    print(training_info.to_string(index=False))
    print()
    
    # Save formatted summary
    summary_file = Path(__file__).parent / 'results' / 'model_comparison_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("MODEL COMPARISON: VALIDATION SET PERFORMANCE\n")
        f.write(f"Dataset: 3,431 sequences (80/20 train/val split)\n")
        f.write("="*120 + "\n\n")
        f.write(val_table.to_string(index=False) + "\n\n")
        f.write(f"Best Validation F1: {best_val['Encoding']} + {best_val['Model']} = {best_val['Val_F1']:.4f}\n\n")
        
        if test_df is not None:
            f.write("="*120 + "\n")
            f.write("MODEL COMPARISON: TEST SET PERFORMANCE\n")
            f.write(f"Dataset: Uniprot_Test_Set.csv (78 sequences)\n")
            f.write("="*120 + "\n\n")
            f.write(test_table.to_string(index=False) + "\n\n")
            f.write(f"Best Test F1: {best_test['Encoding']} + {best_test['Model']} = {best_test['Test_F1']:.4f}\n\n")
        
        f.write("="*120 + "\n")
        f.write("TRAINING INFORMATION\n")
        f.write("="*120 + "\n\n")
        f.write(training_info.to_string(index=False) + "\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    print("\n✓ Comparison complete!")


if __name__ == '__main__':
    main()
