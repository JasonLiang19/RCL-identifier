"""
Metrics for RCL prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)
from typing import Tuple, Dict


MASK_VALUE = 9999


def apply_mask(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mask to remove padded positions.
    
    Args:
        y_true: True labels (batch, max_length, 2)
        y_pred: Predicted labels (batch, max_length, 2)
        
    Returns:
        Tuple of (masked_true, masked_pred)
    """
    # Create mask for valid positions (not padded)
    mask = ~(y_true == MASK_VALUE).all(dim=-1)  # (batch, max_length)
    
    # Flatten and apply mask
    y_true_flat = y_true.reshape(-1, 2)
    y_pred_flat = y_pred.reshape(-1, 2)
    mask_flat = mask.reshape(-1)
    
    y_true_masked = y_true_flat[mask_flat]
    y_pred_masked = y_pred_flat[mask_flat]
    
    return y_true_masked, y_pred_masked


def masked_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate accuracy ignoring masked positions.
    
    Args:
        y_true: True labels (batch, max_length, 2)
        y_pred: Predicted logits (batch, max_length, 2)
        
    Returns:
        Accuracy score
    """
    y_true_masked, y_pred_masked = apply_mask(y_true, y_pred)
    
    if len(y_true_masked) == 0:
        return 0.0
    
    y_true_classes = torch.argmax(y_true_masked, dim=-1)
    y_pred_classes = torch.argmax(y_pred_masked, dim=-1)
    
    accuracy = (y_true_classes == y_pred_classes).float().mean().item()
    return accuracy


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics for RCL prediction.
    
    Args:
        y_true: True labels (N, max_length, 2)
        y_pred: Predicted probabilities (N, max_length, 2)
        
    Returns:
        Dictionary of metrics
    """
    # Flatten and remove masked positions
    y_true_flat = y_true.reshape(-1, 2)
    y_pred_flat = y_pred.reshape(-1, 2)
    
    # Create mask
    mask = ~(y_true_flat == MASK_VALUE).all(axis=-1)
    
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {metric: 0.0 for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']}
    
    # Convert to class indices
    y_true_idx = np.argmax(y_true_clean, axis=-1)
    y_pred_idx = np.argmax(y_pred_clean, axis=-1)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true_idx, y_pred_idx),
        'precision': precision_score(y_true_idx, y_pred_idx, zero_division=0),
        'recall': recall_score(y_true_idx, y_pred_idx, zero_division=0),
        'f1': f1_score(y_true_idx, y_pred_idx, zero_division=0),
        'f1_macro': f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true_idx, y_pred_idx)
    }
    
    # AUC (only if both classes present)
    if len(np.unique(y_true_idx)) > 1:
        try:
            metrics['auc'] = roc_auc_score(y_true_idx, y_pred_clean[:, 1])
        except ValueError:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix ignoring masked positions.
    
    Args:
        y_true: True labels (N, max_length, 2)
        y_pred: Predicted probabilities (N, max_length, 2)
        
    Returns:
        Confusion matrix (2, 2)
    """
    # Flatten and remove masked positions
    y_true_flat = y_true.reshape(-1, 2)
    y_pred_flat = y_pred.reshape(-1, 2)
    
    mask = ~(y_true_flat == MASK_VALUE).all(axis=-1)
    
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    # Convert to class indices
    y_true_idx = np.argmax(y_true_clean, axis=-1)
    y_pred_idx = np.argmax(y_pred_clean, axis=-1)
    
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    
    return cm


def compute_per_protein_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute number of errors per protein.
    
    Args:
        y_true: True labels (N, max_length, 2)
        y_pred: Predicted probabilities (N, max_length, 2)
        
    Returns:
        Array of error counts per protein (N,)
    """
    num_proteins = y_true.shape[0]
    errors = []
    
    for i in range(num_proteins):
        yt = y_true[i]
        yp = y_pred[i]
        
        # Create mask for valid positions
        mask = ~(yt == MASK_VALUE).all(axis=-1)
        
        yt_clean = yt[mask]
        yp_clean = yp[mask]
        
        # Convert to class indices
        yt_idx = np.argmax(yt_clean, axis=-1)
        yp_idx = np.argmax(yp_clean, axis=-1)
        
        # Count errors
        num_errors = np.sum(yt_idx != yp_idx)
        errors.append(num_errors)
    
    return np.array(errors)


class MaskedBCELoss(torch.nn.Module):
    """Binary cross-entropy loss that ignores masked positions."""
    
    def __init__(self, weight: torch.Tensor = None):
        """
        Args:
            weight: Class weights for imbalanced datasets (tensor of shape (2,))
        """
        super().__init__()
        self.weight = weight
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted logits (batch, max_length, 2)
            y_true: True labels (batch, max_length, 2)
            
        Returns:
            Loss value
        """
        # Apply softmax to get probabilities
        y_pred_prob = F.softmax(y_pred, dim=-1)
        
        # Create mask for valid positions
        mask = ~(y_true == MASK_VALUE).all(dim=-1)  # (batch, max_length)
        
        # Compute cross-entropy
        loss = -y_true * torch.log(y_pred_prob + 1e-8)
        
        # Apply class weights if provided
        if self.weight is not None:
            weight_expanded = self.weight.view(1, 1, 2).to(loss.device)
            loss = loss * weight_expanded
        
        # Sum over classes
        loss = loss.sum(dim=-1)  # (batch, max_length)
        
        # Apply mask and compute mean
        loss = loss * mask.float()
        loss = loss.sum() / (mask.sum() + 1e-8)
        
        return loss


class MCCLoss(torch.nn.Module):
    """
    Matthews Correlation Coefficient loss (combined with cross-entropy).
    Similar to the old implementation but in PyTorch.
    """
    
    def __init__(self, weight: torch.Tensor = None, mcc_weight: float = 0.5):
        """
        Args:
            weight: Class weights for cross-entropy
            mcc_weight: Weight for MCC term (0 to 1)
        """
        super().__init__()
        self.bce_loss = MaskedBCELoss(weight)
        self.mcc_weight = mcc_weight
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted logits (batch, max_length, 2)
            y_true: True labels (batch, max_length, 2)
            
        Returns:
            Combined loss value
        """
        # Cross-entropy loss
        ce_loss = self.bce_loss(y_pred, y_true)
        
        # MCC loss (negative MCC to minimize)
        mcc = self.compute_mcc(y_pred, y_true)
        mcc_loss = -mcc
        
        # Combined loss
        loss = (1 - self.mcc_weight) * ce_loss + self.mcc_weight * mcc_loss
        
        return loss
    
    def compute_mcc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Matthews Correlation Coefficient."""
        # Apply softmax
        y_pred_prob = F.softmax(y_pred, dim=-1)
        
        # Create mask
        mask = ~(y_true == MASK_VALUE).all(dim=-1)
        
        # Flatten and apply mask
        y_true_flat = y_true.reshape(-1, 2)[mask.reshape(-1)]
        y_pred_flat = y_pred_prob.reshape(-1, 2)[mask.reshape(-1)]
        
        # Get predictions
        threshold = 0.5
        y_pred_binary = (y_pred_flat[:, 1] > threshold).float()
        y_true_binary = y_true_flat[:, 1]
        
        # Compute TP, TN, FP, FN
        tp = (y_pred_binary * y_true_binary).sum()
        tn = ((1 - y_pred_binary) * (1 - y_true_binary)).sum()
        fp = (y_pred_binary * (1 - y_true_binary)).sum()
        fn = ((1 - y_pred_binary) * y_true_binary).sum()
        
        # Compute MCC
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        mcc = numerator / (denominator + 1e-8)
        
        return mcc
