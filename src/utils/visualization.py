"""
Visualization utilities for RCL prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import csv


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    class_names: list = ["non-RCL", "RCL"]
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (2, 2)
        output_path: Path to save figure
        class_names: Names of classes
    """
    plt.figure(figsize=(8, 6))
    
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        square=True,
        cbar=False,
        xticklabels=[f"Pred {name}" for name in class_names],
        yticklabels=[f"True {name}" for name in class_names]
    )
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    plt.title("Confusion Matrix", pad=20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_histogram(
    errors: np.ndarray,
    output_path: str,
    title: str = "Per-Protein Error Distribution"
):
    """
    Plot histogram of errors per protein.
    
    Args:
        errors: Array of error counts per protein
        output_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    max_errors = int(max(errors))
    bins = np.arange(0, max_errors + 2)
    
    plt.hist(errors, bins=bins, color='skyblue', edgecolor='black', align='left')
    plt.xticks(bins[::max(1, len(bins)//20)], rotation=45, ha="right")
    plt.xlabel("Number of Incorrect Residues per Protein")
    plt.ylabel("Number of Proteins")
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_metrics: dict,
    output_dir: str
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_metrics: Dictionary of validation metrics over epochs
        output_dir: Directory to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss curves
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics curves
    if val_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']
        
        for i, metric in enumerate(metric_names):
            if metric in val_metrics and len(val_metrics[metric]) > 0:
                axes[i].plot(epochs, val_metrics[metric], 'g-', linewidth=2)
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.upper())
                axes[i].set_title(f'Validation {metric.upper()}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metric_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_prediction_example(
    sequence: str,
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    output_path: str,
    protein_id: str = "Example"
):
    """
    Plot prediction example for a single protein.
    
    Args:
        sequence: Protein sequence
        true_labels: True labels (max_length, 2)
        pred_probs: Predicted probabilities (max_length, 2)
        output_path: Path to save figure
        protein_id: Protein identifier
    """
    seq_len = len(sequence)
    positions = np.arange(seq_len)
    
    # Get RCL probabilities
    true_rcl = true_labels[:seq_len, 1]
    pred_rcl = pred_probs[:seq_len, 1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot true labels
    ax1.fill_between(positions, 0, true_rcl, alpha=0.5, color='blue', label='True RCL')
    ax1.set_ylabel('True Label')
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Protein: {protein_id}')
    
    # Plot predictions
    ax2.fill_between(positions, 0, pred_rcl, alpha=0.5, color='red', label='Predicted RCL Prob')
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Threshold')
    ax2.set_xlabel('Residue Position')
    ax2.set_ylabel('Prediction Probability')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_error_histogram_csv(errors: np.ndarray, output_path: str):
    """
    Save error histogram data to CSV.
    
    Args:
        errors: Array of error counts per protein
        output_path: Path to save CSV
    """
    error_values, protein_counts = np.unique(errors, return_counts=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_errors', 'num_proteins'])
        for num_errors, num_proteins in zip(error_values, protein_counts):
            writer.writerow([num_errors, num_proteins])


def plot_rcl_length_distribution(
    predictions: list,
    output_path: str,
    title: str = "Predicted RCL Length Distribution"
):
    """
    Plot distribution of predicted RCL lengths.
    
    Args:
        predictions: List of dictionaries with 'rcl_start' and 'rcl_end'
        output_path: Path to save figure
        title: Plot title
    """
    lengths = []
    for pred in predictions:
        if pred.get('rcl_start') is not None and pred.get('rcl_end') is not None:
            length = pred['rcl_end'] - pred['rcl_start']
            lengths.append(length)
    
    if not lengths:
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('RCL Length (residues)')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
