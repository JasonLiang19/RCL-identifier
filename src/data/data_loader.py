"""
Data loading and encoding utilities for RCL prediction.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from Bio import SeqIO
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class ProteinSequenceDataset(Dataset):
    """PyTorch Dataset for protein sequences with RCL annotations."""
    
    def __init__(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray] = None,
        ids: Optional[List[str]] = None,
        max_length: int = 1024
    ):
        """
        Args:
            sequences: List of protein sequences
            labels: Optional labels (N, max_length, 2) for binary classification
            ids: Optional sequence identifiers
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.ids = ids if ids is not None else [f"seq_{i}" for i in range(len(sequences))]
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {
            'sequence': self.sequences[idx],
            'id': self.ids[idx],
            'length': min(len(self.sequences[idx]), self.max_length)
        }
        
        if self.labels is not None:
            item['label'] = torch.from_numpy(self.labels[idx]).float()
            
        return item


def read_fasta(fasta_path: str) -> Tuple[List[str], List[str]]:
    """
    Read protein sequences from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Tuple of (sequence_ids, sequences)
    """
    sequences = []
    ids = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
        
    return ids, sequences


def read_csv_with_annotations(
    csv_path: str,
    max_length: int = 1024
) -> Tuple[List[str], np.ndarray, List[str], List[str]]:
    """
    Read CSV file with RCL annotations.
    
    Args:
        csv_path: Path to CSV file with columns: id, Sequence, rcl_start, rcl_end
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (ids, labels, sequences, rcl_sequences)
    """
    df = pd.read_csv(csv_path)
    
    # Filter by length
    df = df[df["Sequence"].str.len() <= max_length].copy()
    
    ids = []
    sequences = []
    labels = []
    rcl_seqs = []
    skipped_count = 0
    
    for _, row in df.iterrows():
        protein_id = str(row["id"]).strip()
        sequence = str(row["Sequence"]).strip()
        
        # Check if RCL annotation exists and is valid
        rcl_start_val = row.get('rcl_start')
        rcl_end_val = row.get('rcl_end')
        
        # Determine if this sequence has RCL annotation columns
        has_annotation_columns = ('rcl_start' in row.index and 'rcl_end' in row.index)
        
        # Check for valid RCL annotation
        has_valid_annotation = False
        rcl_start = None
        rcl_end = None
        
        if has_annotation_columns:
            # For sequences with annotation columns (e.g., serpins)
            if (pd.notna(rcl_start_val) and pd.notna(rcl_end_val) and 
                str(rcl_start_val).lower() not in ['no result', 'no start', '#value!', 'nan'] and
                str(rcl_end_val).lower() not in ['no result', 'no end', '#value!', 'nan']):
                try:
                    rcl_start = int(rcl_start_val) - 1  # Convert to 0-indexed
                    rcl_end = int(rcl_end_val)
                    
                    # Validate indices
                    if 0 <= rcl_start < rcl_end <= len(sequence):
                        has_valid_annotation = True
                except (ValueError, TypeError):
                    # Invalid integer conversion
                    pass
            
            # Skip serpins without valid annotations (these are incomplete/bad data)
            if not has_valid_annotation:
                skipped_count += 1
                continue
        # else: Non-serpin sequences (no annotation columns) are valid with all non-RCL labels
        
        # Create binary labels: [non-RCL, RCL]
        label = np.zeros((max_length, 2), dtype=np.float32)
        label[:, 0] = 1  # Initialize as non-RCL
        
        # Mark RCL region if valid annotation exists
        if has_valid_annotation and rcl_start is not None and rcl_end is not None:
            for i in range(rcl_start, min(rcl_end, len(sequence))):
                label[i] = [0, 1]
        
        rcl_seq = sequence[rcl_start:rcl_end] if (has_valid_annotation and rcl_start is not None) else ""
        
        # Mask positions beyond sequence length
        for i in range(len(sequence), max_length):
            label[i] = [9999, 9999]  # Masked value
            
        ids.append(protein_id)
        sequences.append(sequence)
        labels.append(label)
        rcl_seqs.append(rcl_seq)
    
    if skipped_count > 0:
        print(f"  ⚠ Skipped {skipped_count} serpin sequences with invalid/missing RCL annotations")
        
    return ids, np.array(labels), sequences, rcl_seqs


def parse_fasta_with_rcl_annotations(
    fasta_path: str,
    max_length: int = 1024
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Parse FASTA file with RCL annotations in header (e.g., '>ID rcl:350-370').
    
    Args:
        fasta_path: Path to FASTA file
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (ids, labels, sequences)
    """
    ids = []
    sequences = []
    labels = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)
        
        # Skip if too long
        if len(sequence) > max_length:
            continue
            
        # Parse RCL annotation from description
        label = np.zeros((max_length, 2), dtype=np.float32)
        label[:, 0] = 1  # Initialize as non-RCL
        
        # Look for rcl:start-end pattern
        desc = record.description
        if 'rcl:' in desc:
            try:
                rcl_part = desc.split('rcl:')[1].split()[0]
                if '-' in rcl_part and rcl_part != 'no result-nan':
                    start, end = rcl_part.split('-')
                    rcl_start = int(start) - 1  # Convert to 0-indexed
                    rcl_end = int(end)
                    
                    # Mark RCL region
                    for i in range(rcl_start, min(rcl_end, len(sequence))):
                        label[i] = [0, 1]
            except (ValueError, IndexError):
                # No valid RCL annotation, keep as all non-RCL
                pass
        
        # Mask positions beyond sequence length
        for i in range(len(sequence), max_length):
            label[i] = [9999, 9999]
            
        ids.append(seq_id)
        sequences.append(sequence)
        labels.append(label)
        
    return ids, np.array(labels), sequences


def load_encoding_matrix(encoding_type: str, encoding_dir: str = "data/encodings") -> Dict:
    """
    Load encoding matrix from JSON file.
    
    Args:
        encoding_type: Type of encoding (onehot, blosum)
        encoding_dir: Directory containing encoding JSON files
        
    Returns:
        Dictionary mapping amino acids to vectors
    """
    encoding_files = {
        'onehot': 'One_hot.json',
        'blosum': 'BLOSUM62.json'
    }
    
    if encoding_type not in encoding_files:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
        
    encoding_path = Path(encoding_dir) / encoding_files[encoding_type]
    
    with open(encoding_path, 'r') as f:
        encoding_map = json.load(f)
        
    return encoding_map


def create_data_loaders(
    train_sequences: List[str],
    train_labels: np.ndarray,
    val_sequences: List[str],
    val_labels: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = 1024
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch data loaders for training and validation.
    
    Args:
        train_sequences: Training sequences
        train_labels: Training labels
        val_sequences: Validation sequences
        val_labels: Validation labels
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ProteinSequenceDataset(
        train_sequences, train_labels, max_length=max_length
    )
    val_dataset = ProteinSequenceDataset(
        val_sequences, val_labels, max_length=max_length
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
