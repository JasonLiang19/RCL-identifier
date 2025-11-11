"""
Protein sequence encoding utilities.
"""

import json
import numpy as np
import torch
from typing import List, Dict, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Try to import ESM library for native ESM2 support
try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


class SequenceEncoder:
    """Base class for sequence encoders."""
    
    def __init__(self, max_length: int = 1024):
        self.max_length = max_length
        
    def encode(self, sequence: str) -> np.ndarray:
        """Encode a single sequence."""
        raise NotImplementedError
        
    def encode_batch(self, sequences: List[str]) -> np.ndarray:
        """Encode a batch of sequences."""
        encoded = [self.encode(seq) for seq in tqdm(sequences, desc="Encoding sequences")]
        return np.array(encoded, dtype=np.float32)
    
    def get_encoding_dim(self) -> int:
        """Get the dimension of the encoding."""
        raise NotImplementedError


class OneHotEncoder(SequenceEncoder):
    """One-hot encoding for protein sequences."""
    
    def __init__(self, max_length: int = 1024, encoding_path: Optional[str] = None):
        super().__init__(max_length)
        
        if encoding_path is None:
            encoding_path = "data/encodings/One_hot.json"
            
        with open(encoding_path, 'r') as f:
            encoding_data = json.load(f)
            
        # Extract metadata if present
        if 'dimension' in encoding_data:
            self.dim = encoding_data['dimension']
        else:
            # Fallback: get dimension from first amino acid entry
            for key, value in encoding_data.items():
                if isinstance(value, list):
                    self.dim = len(value)
                    break
        
        # Build encoding map (only amino acid entries)
        self.encoding_map = {
            key: value for key, value in encoding_data.items()
            if isinstance(value, list)
        }
        
    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode sequence using one-hot encoding.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Encoded sequence of shape (max_length, dim)
        """
        # Initialize output array
        encoded = np.zeros((self.max_length, self.dim), dtype=np.float32)
        
        # Encode each residue up to max_length
        seq_len = min(len(sequence), self.max_length)
        for i, aa in enumerate(sequence[:seq_len]):
            encoding = self.encoding_map.get(aa, self.encoding_map.get('X', [0] * self.dim))
            encoded[i] = encoding
            
        return encoded
    
    def get_encoding_dim(self) -> int:
        return self.dim


class BLOSUMEncoder(SequenceEncoder):
    """BLOSUM62 matrix encoding for protein sequences."""
    
    def __init__(self, max_length: int = 1024, encoding_path: Optional[str] = None):
        super().__init__(max_length)
        
        if encoding_path is None:
            encoding_path = "data/encodings/BLOSUM62.json"
            
        with open(encoding_path, 'r') as f:
            encoding_data = json.load(f)
            
        # Extract metadata if present
        if 'dimension' in encoding_data:
            self.dim = encoding_data['dimension']
        else:
            # Fallback: get dimension from first amino acid entry
            for key, value in encoding_data.items():
                if isinstance(value, list):
                    self.dim = len(value)
                    break
        
        # Build encoding map (only amino acid entries)
        self.encoding_map = {
            key: value for key, value in encoding_data.items()
            if isinstance(value, list)
        }
        
    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode sequence using BLOSUM62 matrix.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Encoded sequence of shape (max_length, dim)
        """
        # Initialize output array
        encoded = np.zeros((self.max_length, self.dim), dtype=np.float32)
        
        # Encode each residue up to max_length
        seq_len = min(len(sequence), self.max_length)
        for i, aa in enumerate(sequence[:seq_len]):
            encoding = self.encoding_map.get(aa, self.encoding_map.get('X', [0] * self.dim))
            encoded[i] = encoding
            
        return encoded
    
    def get_encoding_dim(self) -> int:
        return self.dim


class ESM2Encoder(SequenceEncoder):
    """ESM2 protein language model encoding."""
    
    def __init__(
        self,
        max_length: int = 1024,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "cuda"
    ):
        super().__init__(max_length)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.dim = self.model.config.hidden_size
        
    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode sequence using ESM2 model.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Encoded sequence of shape (max_length, dim)
        """
        # Truncate if needed
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
            
        # Tokenize with spaces between amino acids
        spaced_seq = " ".join(sequence)
        tokens = self.tokenizer(spaced_seq, return_tensors="pt", add_special_tokens=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state[0, 1:-1]  # Remove [CLS] and [SEP] tokens
            
        # Convert to numpy
        encoded = embeddings.cpu().numpy()
        
        # Pad to max_length
        if len(encoded) < self.max_length:
            padding = np.zeros((self.max_length - len(encoded), self.dim), dtype=np.float32)
            encoded = np.vstack([encoded, padding])
            
        return encoded
    
    def encode_batch(self, sequences: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Encode a batch of sequences (more efficient for ESM2).
        
        Args:
            sequences: List of protein sequences
            batch_size: Number of sequences to process at once
            
        Returns:
            Encoded sequences of shape (n_sequences, max_length, dim)
        """
        all_encoded = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding with ESM2"):
            batch = sequences[i:i+batch_size]
            batch_encoded = [self.encode(seq) for seq in batch]
            all_encoded.extend(batch_encoded)
            
        return np.array(all_encoded, dtype=np.float32)
    
    def get_encoding_dim(self) -> int:
        return self.dim


class ESM2_650M_Encoder(SequenceEncoder):
    """ESM2 650M protein language model encoding using native ESM library."""
    
    def __init__(
        self,
        max_length: int = 1024,
        device: str = "cuda"
    ):
        super().__init__(max_length)
        
        if not ESM_AVAILABLE:
            raise ImportError(
                "ESM library not found. Install with: pip install fair-esm"
            )
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load ESM2 650M model using native ESM library
        print("Loading ESM2 650M model...")
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        # ESM2 650M has 1280-dimensional embeddings (layer 33)
        self.dim = 1280
        self.repr_layer = 33
        
    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode sequence using ESM2 650M model.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Encoded sequence of shape (max_length, dim)
        """
        # Truncate if needed
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
            
        # Prepare data for batch converter
        data = [("seq", sequence)]
        _, _, toks = self.batch_converter(data)
        
        if torch.cuda.is_available():
            toks = toks.to(self.device)
        
        # Get embeddings from layer 33
        with torch.no_grad():
            results = self.model(toks, repr_layers=[self.repr_layer])
            # Shape: [batch=1, length+2, dim] (includes BOS and EOS tokens)
            embeddings = results["representations"][self.repr_layer][0, 1:-1]  # Remove BOS/EOS
            
        # Convert to numpy
        encoded = embeddings.cpu().numpy()
        
        # Pad to max_length
        output = np.zeros((self.max_length, self.dim), dtype=np.float32)
        seq_len = min(len(encoded), self.max_length)
        output[:seq_len] = encoded[:seq_len]
            
        return output
    
    def encode_batch(self, sequences: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Encode a batch of sequences (more efficient for ESM2).
        
        Args:
            sequences: List of protein sequences
            batch_size: Number of sequences to process at once (default=4, use 1-2 for large datasets)
            
        Returns:
            Encoded sequences of shape (n_sequences, max_length, dim)
        """
        all_encoded = []
        
        # Use smaller batch size to avoid OOM
        effective_batch_size = min(batch_size, 2)  # Limit to 2 sequences at a time
        
        for i in tqdm(range(0, len(sequences), effective_batch_size), desc="Encoding with ESM2 650M"):
            # Prepare batch
            batch_seqs = sequences[i:i+effective_batch_size]
            batch_data = [(f"seq_{j}", seq[:self.max_length]) for j, seq in enumerate(batch_seqs)]
            
            _, _, toks = self.batch_converter(batch_data)
            
            if torch.cuda.is_available():
                toks = toks.to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                results = self.model(toks, repr_layers=[self.repr_layer])
                embeddings = results["representations"][self.repr_layer][:, 1:-1]  # Remove BOS/EOS
                
            # Process each sequence in batch
            for j, seq in enumerate(batch_seqs):
                seq_embeddings = embeddings[j].cpu().numpy()
                
                # Pad to max_length
                output = np.zeros((self.max_length, self.dim), dtype=np.float32)
                seq_len = min(len(seq_embeddings), self.max_length)
                output[:seq_len] = seq_embeddings[:seq_len]
                
                all_encoded.append(output)
            
            # Clear GPU cache periodically to avoid OOM
            if torch.cuda.is_available() and i % 100 == 0:
                torch.cuda.empty_cache()
            
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return np.array(all_encoded, dtype=np.float32)
    
    def get_encoding_dim(self) -> int:
        return self.dim


def get_encoder(
    encoding_type: str,
    max_length: int = 1024,
    device: str = "cuda",
    encoding_dir: str = "data/encodings"
) -> SequenceEncoder:
    """
    Factory function to get the appropriate encoder.
    
    Args:
        encoding_type: Type of encoding (onehot, blosum, esm2, esm2_650m)
        max_length: Maximum sequence length
        device: Device for ESM2 model (cuda/cpu)
        encoding_dir: Directory containing encoding files
        
    Returns:
        SequenceEncoder instance
    """
    encoding_type = encoding_type.lower()
    
    if encoding_type == "onehot":
        return OneHotEncoder(
            max_length=max_length,
            encoding_path=f"{encoding_dir}/One_hot.json"
        )
    elif encoding_type == "blosum":
        return BLOSUMEncoder(
            max_length=max_length,
            encoding_path=f"{encoding_dir}/BLOSUM62.json"
        )
    elif encoding_type == "esm2":
        return ESM2Encoder(
            max_length=max_length,
            device=device
        )
    elif encoding_type == "esm2_650m":
        return ESM2_650M_Encoder(
            max_length=max_length,
            device=device
        )
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Choose from: onehot, blosum, esm2, esm2_650m")


class EncodedDataset(torch.utils.data.Dataset):
    """Dataset that holds pre-encoded sequences.
    
    Supports both regular numpy arrays and memory-mapped arrays.
    For memory-mapped arrays, conversion to tensor happens lazily in __getitem__.
    """
    
    def __init__(
        self,
        encodings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        ids: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None,
        indices: Optional[np.ndarray] = None,
        lazy_load: bool = False
    ):
        """
        Args:
            encodings: Encoded sequences (N, max_length, encoding_dim)
            labels: Optional labels (N, max_length, 2)
            ids: Optional sequence IDs
            sequences: Optional raw sequences
            indices: Optional indices to subset the data (for train/val split)
            lazy_load: If True, keep as numpy and convert to tensor in __getitem__ (for memory-mapped arrays)
        """
        self.lazy_load = lazy_load
        self.indices = indices if indices is not None else np.arange(len(encodings))
        
        if lazy_load:
            # Keep as numpy arrays for memory-mapped data
            self.encodings = encodings
            self.labels = labels
        else:
            # Convert to tensors immediately for regular arrays
            self.encodings = torch.from_numpy(encodings).float()
            self.labels = torch.from_numpy(labels).float() if labels is not None else None
        
        self.ids = ids if ids is not None else [f"seq_{i}" for i in range(len(encodings))]
        self.sequences = sequences
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map to actual index in the full dataset
        actual_idx = self.indices[idx]
        
        if self.lazy_load:
            # Convert to tensor on-the-fly for memory-mapped arrays
            encoding = torch.from_numpy(self.encodings[actual_idx]).float()
            label = torch.from_numpy(self.labels[actual_idx]).float() if self.labels is not None else None
        else:
            # Already tensors
            encoding = self.encodings[actual_idx]
            label = self.labels[actual_idx] if self.labels is not None else None
        
        item = {
            'encoding': encoding,
            'id': self.ids[actual_idx]
        }
        
        if label is not None:
            item['label'] = label
            
        if self.sequences is not None:
            item['sequence'] = self.sequences[actual_idx]
            
        return item
