"""
Pre-compute and save ESM2 embeddings to avoid OOM during training.
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys

from data import read_csv_with_annotations, get_encoder


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-compute ESM2 embeddings')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--encoding', type=str, default='esm2_650m',
                       choices=['esm2', 'esm2_650m'],
                       help='Encoding type')
    parser.add_argument('--output-dir', type=str, default='data/embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for encoding (use 1 for large proteins)')
    parser.add_argument('--checkpoint-every', type=int, default=500,
                       help='Save checkpoint every N sequences')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    
    return parser.parse_args()


def save_checkpoint(output_file, encodings, labels, ids, sequences, last_idx):
    """Save a checkpoint."""
    checkpoint_file = output_file.parent / f'{output_file.stem}_checkpoint.npz'
    np.savez_compressed(
        checkpoint_file,
        encodings=np.array(encodings, dtype=np.float32),
        labels=labels[:last_idx+1],
        ids=ids[:last_idx+1],
        sequences=sequences[:last_idx+1],
        last_idx=last_idx
    )
    print(f"Checkpoint saved at index {last_idx}")


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading data...")
    
    # Load serpin and non-serpin data
    serpin_ids, serpin_labels, serpin_seqs, _ = read_csv_with_annotations(
        config['data']['train_serpin'],
        max_length=config['data']['max_length']
    )
    
    non_serpin_ids, non_serpin_labels, non_serpin_seqs, _ = read_csv_with_annotations(
        config['data']['train_non_serpin'],
        max_length=config['data']['max_length']
    )
    
    # Combine datasets
    all_ids = serpin_ids + non_serpin_ids
    all_seqs = serpin_seqs + non_serpin_seqs
    all_labels = np.vstack([serpin_labels, non_serpin_labels])
    
    print(f"Total sequences: {len(all_seqs)}")
    print(f"Serpin sequences: {len(serpin_seqs)}")
    print(f"Non-serpin sequences: {len(non_serpin_seqs)}")
    
    # Check for checkpoint
    output_file = output_dir / f'{args.encoding}_embeddings.npz'
    checkpoint_file = output_dir / f'{args.encoding}_embeddings_checkpoint.npz'
    
    start_idx = 0
    all_encodings = []
    
    if args.resume and checkpoint_file.exists():
        print(f"\nResuming from checkpoint: {checkpoint_file}")
        checkpoint = np.load(checkpoint_file, allow_pickle=True)
        all_encodings = list(checkpoint['encodings'])
        start_idx = int(checkpoint['last_idx']) + 1
        print(f"Resuming from index {start_idx}")
    
    # Initialize encoder
    print(f"\nInitializing {args.encoding} encoder...")
    encoder = get_encoder(
        args.encoding,
        max_length=config['data']['max_length'],
        device=device
    )
    
    # Encode sequences one at a time
    print(f"\nEncoding sequences starting from index {start_idx}...")
    print(f"Checkpoints will be saved every {args.checkpoint_every} sequences")
    print("This may take a while... (Ctrl+C to stop and resume later)")
    
    try:
        for i in tqdm(range(start_idx, len(all_seqs)), desc="Encoding sequences", initial=start_idx, total=len(all_seqs)):
            seq = all_seqs[i]
            encoding = encoder.encode(seq)
            all_encodings.append(encoding)
            
            # Clear cache periodically
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save checkpoint periodically
            if (i + 1) % args.checkpoint_every == 0:
                save_checkpoint(output_file, all_encodings, all_labels, all_ids, all_seqs, i)
        
        print(f"\n✓ Encoding complete!")
        encodings = np.array(all_encodings, dtype=np.float32)
        print(f"Encoding shape: {encodings.shape}")
        
        # Save final file
        print(f"\nSaving final embeddings to: {output_file}")
        np.savez_compressed(
            output_file,
            encodings=encodings,
            labels=all_labels,
            ids=all_ids,
            sequences=all_seqs
        )
        
        print(f"✓ Saved {len(encodings)} embeddings")
        print(f"File size: {output_file.stat().st_size / 1e9:.2f} GB")
        
        # Remove checkpoint file
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("✓ Checkpoint file removed")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted! Saving checkpoint...")
        if len(all_encodings) > 0:
            save_checkpoint(output_file, all_encodings, all_labels, all_ids, all_seqs, len(all_encodings) - 1)
            print(f"✓ Progress saved. Resume with: python {' '.join(sys.argv)} --resume")
        sys.exit(0)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n✓ Done! Use these embeddings with:")
    print(f"  python src/train.py --precomputed {output_file} --model unet --epochs 50 --batch-size 32 --multi-gpu")


if __name__ == '__main__':
    main()
