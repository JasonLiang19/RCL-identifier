#!/usr/bin/env python3
"""
RCL Prediction Script

Predicts RCL locations in protein sequences using trained models.
Supports three U-Net models: OneHot, BLOSUM62, and ESM2_650M encodings.

Usage:
  python predict.py input.fasta --model onehot_unet
  python predict.py input.fasta --model blosum_unet
  python predict.py input.fasta --model esm2_650m_unet -seq_fasta -RCL_fasta
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.encoders import OneHotEncoder, BLOSUMEncoder, ESM2_650M_Encoder, EncodedDataset
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict RCL locations in protein sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - outputs TSV with RCL locations
  python predict.py input/proteins.fasta --model onehot_unet
  python predict.py input/proteins.fasta --model blosum_unet
  python predict.py input/proteins.fasta --model esm2_650m_unet
  
  # Also generate FASTA files
  python predict.py input/proteins.fasta --model esm2_650m_unet -seq_fasta -RCL_fasta
  
  # Use custom output directory and filtering
  python predict.py input/proteins.fasta --model blosum_unet --output-dir results/ --min-consecutive 12
  
  # Use different threshold
  python predict.py input/proteins.fasta --model esm2_650m_unet --threshold 0.7 --min-rcl-length 10
"""
    )
    
    parser.add_argument('input', type=str,
                       help='Input FASTA file with protein sequences')
    parser.add_argument('--model', '-m', type=str, required=True,
                       choices=['onehot_unet', 'blosum_unet', 'esm2_650m_unet'],
                       help='Model to use for prediction')
    parser.add_argument('--output-dir', '-o', type=str, default='prediction/output',
                       help='Output directory (default: prediction/output/)')
    parser.add_argument('--min-rcl-length', type=int, default=7,
                       help='Minimum number of amino acids to identify as RCL (default: 7)')
    parser.add_argument('--min-consecutive', type=int, default=12,
                       help='Minimum consecutive residues for valid RCL region (default: 12)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for RCL prediction (default: 0.5)')
    parser.add_argument('-seq_fasta', action='store_true',
                       help='Generate FASTA file with all sequences identified as having RCL')
    parser.add_argument('-RCL_fasta', action='store_true',
                       help='Generate FASTA file with only RCL sequences extracted')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for prediction (default: 8)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Process sequences in chunks to save memory (default: 10000)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    
    return parser.parse_args()


def read_fasta(fasta_file):
    """
    Read FASTA file and extract sequences.
    
    Returns:
        List of tuples: [(accession, full_header, sequence), ...]
    """
    sequences = []
    current_accession = None
    current_header = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_accession is not None:
                    sequences.append((
                        current_accession,
                        current_header,
                        ''.join(current_seq)
                    ))
                
                # Parse new header
                current_header = line[1:].strip()  # Remove '>'
                # Accession is the first part before space
                current_accession = current_header.split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget the last sequence
        if current_accession is not None:
            sequences.append((
                current_accession,
                current_header,
                ''.join(current_seq)
            ))
    
    return sequences


def load_model_and_encoder(model_name, device):
    """
    Load trained model and appropriate encoder.
    
    Returns:
        (model, encoder, config)
    """
    # Determine model directory
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'analysis' / 'results' / model_name
    
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Available models should be in: {project_root / 'analysis' / 'results'}/"
        )
    
    # Load config
    config_file = model_dir / 'config.json'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Determine encoding and dimension
    # Get project root for encoding files
    project_root = Path(__file__).parent.parent
    
    if 'onehot' in model_name:
        encoding_type = 'onehot'
        encoding_dim = 21
        max_length = config['data']['max_length']
        encoding_path = str(project_root / 'data' / 'encodings' / 'One_hot.json')
        encoder = OneHotEncoder(max_length=max_length, encoding_path=encoding_path)
    elif 'blosum' in model_name:
        encoding_type = 'blosum'
        encoding_dim = 20
        max_length = config['data']['max_length']
        encoding_path = str(project_root / 'data' / 'encodings' / 'BLOSUM62.json')
        encoder = BLOSUMEncoder(max_length=max_length, encoding_path=encoding_path)
    elif 'esm2_650m' in model_name:
        encoding_type = 'esm2_650m'
        encoding_dim = 1280
        max_length = config['data']['max_length']
        encoder = ESM2_650M_Encoder(max_length=max_length, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    
    # Create model
    model_type = 'unet'  # Both specified models use U-Net
    model = get_model(
        model_type,
        encoding_dim,
        config['models'][model_type],
        num_classes=2
    )
    
    # Load weights
    checkpoint_file = model_dir / 'best_model.pt'
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Handle DataParallel wrapped models
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
    
    return model, encoder, config


def extract_rcl_region(pred_probs, seq_length, threshold=0.5, min_consecutive=12):
    """
    Extract RCL start/end positions from predictions.
    Filters regions by minimum consecutive residues to reduce false positives.
    
    Args:
        pred_probs: Prediction probabilities (max_length, 2)
        seq_length: Actual sequence length
        threshold: Probability threshold
        min_consecutive: Minimum consecutive residues required for valid RCL region
        
    Returns:
        Tuple of (start, end, rcl_length, confidence)
        Positions are 0-indexed. Returns (None, None, 0, 0.0) if no RCL found.
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


def predict_sequences(model, encoder, sequences, device, batch_size=8):
    """
    Encode sequences and make predictions.
    
    Returns:
        numpy array of prediction probabilities (N, max_length, 2)
    """
    print("\nEncoding sequences...")
    
    # Extract just the sequence strings
    seq_strings = [seq for _, _, seq in sequences]
    
    # Encode sequences
    if isinstance(encoder, ESM2_650M_Encoder):
        # ESM2 encoder has its own batching
        encodings = encoder.encode_batch(seq_strings, batch_size=1)
    else:
        # BLOSUM encoder
        encodings = encoder.encode_batch(seq_strings)
    
    # Create labels placeholder (not used but needed for dataset)
    labels = np.zeros((len(encodings), encoder.max_length, 2))
    
    # Create dataset
    dataset = EncodedDataset(
        encodings,
        labels,
        ids=[acc for acc, _, _ in sequences],
        sequences=seq_strings,
        lazy_load=False  # Already in memory as regular arrays
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Single process for prediction
    )
    
    # Make predictions
    print("\nMaking predictions...")
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            encodings_batch = batch['encoding'].to(device)
            
            # Forward pass
            outputs = model(encodings_batch)
            probs = torch.softmax(outputs, dim=-1)
            
            all_preds.append(probs.cpu().numpy())
    
    return np.concatenate(all_preds, axis=0)


def predict_sequences_chunked(model, encoder, sequences, device, batch_size=8, chunk_size=10000):
    """
    Process sequences in chunks to avoid memory issues.
    Yields results for each chunk.
    
    Yields:
        (chunk_predictions, chunk_start_idx)
    """
    total_sequences = len(sequences)
    num_chunks = (total_sequences + chunk_size - 1) // chunk_size
    
    print(f"\nProcessing {total_sequences} sequences in {num_chunks} chunks of {chunk_size}...")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_sequences)
        chunk_sequences = sequences[start_idx:end_idx]
        
        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} (sequences {start_idx + 1}-{end_idx}) ---")
        
        # Extract sequence strings
        seq_strings = [seq for _, _, seq in chunk_sequences]
        
        # Encode sequences
        print("Encoding sequences...")
        if isinstance(encoder, ESM2_650M_Encoder):
            encodings = encoder.encode_batch(seq_strings, batch_size=1)
        else:
            encodings = encoder.encode_batch(seq_strings)
        
        # Create labels placeholder
        labels = np.zeros((len(encodings), encoder.max_length, 2))
        
        # Create dataset
        dataset = EncodedDataset(
            encodings,
            labels,
            ids=[acc for acc, _, _ in chunk_sequences],
            sequences=seq_strings,
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
        print("Making predictions...")
        chunk_preds = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f'Chunk {chunk_idx + 1}/{num_chunks}'):
                encodings_batch = batch['encoding'].to(device)
                
                # Forward pass
                outputs = model(encodings_batch)
                probs = torch.softmax(outputs, dim=-1)
                
                chunk_preds.append(probs.cpu().numpy())
        
        chunk_preds = np.concatenate(chunk_preds, axis=0)
        
        # Free memory
        del encodings, labels, dataset, data_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        yield chunk_preds, start_idx


def write_tsv_output(results, output_file):
    """
    Write TSV file with RCL locations.
    
    Format:
    Accession   RCL_start_position   RCL_end_position
    """
    with open(output_file, 'w') as f:
        # Header
        f.write("Accession\tRCL_start_position\tRCL_end_position\n")
        
        # Data rows
        for result in results:
            if result['has_rcl']:
                f.write(f"{result['accession']}\t{result['rcl_start']}\t{result['rcl_end']}\n")


def write_seq_fasta(results, output_file):
    """
    Write FASTA file with sequences identified as having RCL.
    
    Format:
    >Accession
    SEQUENCE
    """
    with open(output_file, 'w') as f:
        for result in results:
            if result['has_rcl']:
                f.write(f">{result['accession']}\n")
                # Write sequence in lines of 80 characters
                seq = result['sequence']
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + '\n')


def write_rcl_fasta(results, output_file):
    """
    Write FASTA file with only RCL sequences.
    
    Format:
    >Accession RCL {Start} - {End}
    RCL_SEQUENCE
    """
    with open(output_file, 'w') as f:
        for result in results:
            if result['has_rcl']:
                f.write(f">{result['accession']} RCL {result['rcl_start']} - {result['rcl_end']}\n")
                # Write RCL sequence in lines of 80 characters
                rcl_seq = result['rcl_sequence']
                for i in range(0, len(rcl_seq), 80):
                    f.write(rcl_seq[i:i+80] + '\n')


def main():
    args = parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("RCL PREDICTION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Minimum RCL length: {args.min_rcl_length}")
    print(f"Threshold: {args.threshold}")
    
    # Load model and encoder
    print(f"\nLoading model...")
    model, encoder, config = load_model_and_encoder(args.model, device)
    print(f"✓ Model loaded successfully")
    
    # Read input sequences
    print(f"\nReading sequences from: {args.input}")
    sequences = read_fasta(args.input)
    print(f"✓ Loaded {len(sequences)} sequences")
    
    # Filter by length
    max_length = encoder.max_length
    valid_sequences = []
    skipped = 0
    
    for acc, header, seq in sequences:
        if len(seq) <= max_length:
            valid_sequences.append((acc, header, seq))
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"⚠ Skipped {skipped} sequences exceeding max length ({max_length})")
    
    sequences = valid_sequences
    
    if len(sequences) == 0:
        print("Error: No valid sequences to process!")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file basename from input
    input_basename = Path(args.input).stem
    
    # Extract encoding name from model name (e.g., 'onehot_unet' -> 'onehot')
    encoding_name = args.model.replace('_unet', '').replace('_cnn', '')
    
    # Append encoding and threshold to filename
    threshold_str = f"_thr{args.threshold:.2f}".replace('.', 'p')
    
    # Determine output files (with encoding name)
    tsv_file = output_dir / f"{input_basename}_{encoding_name}_RCL_predictions{threshold_str}.tsv"
    seq_fasta_file = output_dir / f"{input_basename}_{encoding_name}_with_RCL{threshold_str}.fasta" if args.seq_fasta else None
    rcl_fasta_file = output_dir / f"{input_basename}_{encoding_name}_RCL_only{threshold_str}.fasta" if args.RCL_fasta else None
    
    # Open output files
    tsv_out = open(tsv_file, 'w')
    tsv_out.write("Accession\tRCL_start_position\tRCL_end_position\n")
    
    seq_fasta_out = open(seq_fasta_file, 'w') if seq_fasta_file else None
    rcl_fasta_out = open(rcl_fasta_file, 'w') if rcl_fasta_file else None
    
    # Track statistics
    num_with_rcl = 0
    total_confidence = 0.0
    total_rcl_length = 0.0
    
    # Process in chunks for large datasets
    use_chunked = len(sequences) > args.chunk_size
    
    if use_chunked:
        # Chunked processing for memory efficiency
        for chunk_preds, chunk_start_idx in predict_sequences_chunked(
            model, encoder, sequences, device, args.batch_size, args.chunk_size
        ):
            # Extract RCL regions for this chunk
            print("Extracting RCL regions for chunk...")
            
            for i in range(len(chunk_preds)):
                seq_idx = chunk_start_idx + i
                accession, header, sequence = sequences[seq_idx]
                pred = chunk_preds[i]
                
                # Extract RCL (0-indexed)
                rcl_start_0, rcl_end_0, rcl_length, confidence = extract_rcl_region(
                    pred,
                    len(sequence),
                    threshold=args.threshold,
                    min_consecutive=args.min_consecutive
                )
                
                # Check if meets minimum length requirement
                has_rcl = (rcl_start_0 is not None and rcl_length >= args.min_rcl_length)
                
                if has_rcl:
                    num_with_rcl += 1
                    total_confidence += confidence
                    total_rcl_length += rcl_length
                    
                    # Convert to 1-indexed for output
                    rcl_start_1 = rcl_start_0 + 1
                    rcl_end_1 = rcl_end_0
                    rcl_sequence = sequence[rcl_start_0:rcl_end_0]
                    
                    # Write to TSV
                    tsv_out.write(f"{accession}\t{rcl_start_1}\t{rcl_end_1}\n")
                    
                    # Write to seq_fasta if requested
                    if seq_fasta_out:
                        seq_fasta_out.write(f">{accession}\n")
                        for j in range(0, len(sequence), 80):
                            seq_fasta_out.write(sequence[j:j+80] + '\n')
                    
                    # Write to RCL_fasta if requested
                    if rcl_fasta_out:
                        rcl_fasta_out.write(f">{accession} RCL {rcl_start_1} - {rcl_end_1}\n")
                        for j in range(0, len(rcl_sequence), 80):
                            rcl_fasta_out.write(rcl_sequence[j:j+80] + '\n')
            
            # Free memory
            del chunk_preds
    
    else:
        # Small dataset - process all at once
        pred_probs = predict_sequences(model, encoder, sequences, device, args.batch_size)
        
        # Extract RCL regions
        print("\nExtracting RCL regions...")
        
        for i, (accession, header, sequence) in enumerate(sequences):
            pred = pred_probs[i]
            
            # Extract RCL (0-indexed)
            rcl_start_0, rcl_end_0, rcl_length, confidence = extract_rcl_region(
                pred,
                len(sequence),
                threshold=args.threshold,
                min_consecutive=args.min_consecutive
            )
            
            # Check if meets minimum length requirement
            has_rcl = (rcl_start_0 is not None and rcl_length >= args.min_rcl_length)
            
            if has_rcl:
                num_with_rcl += 1
                total_confidence += confidence
                total_rcl_length += rcl_length
                
                # Convert to 1-indexed for output
                rcl_start_1 = rcl_start_0 + 1
                rcl_end_1 = rcl_end_0
                rcl_sequence = sequence[rcl_start_0:rcl_end_0]
                
                # Write to TSV
                tsv_out.write(f"{accession}\t{rcl_start_1}\t{rcl_end_1}\n")
                
                # Write to seq_fasta if requested
                if seq_fasta_out:
                    seq_fasta_out.write(f">{accession}\n")
                    for j in range(0, len(sequence), 80):
                        seq_fasta_out.write(sequence[j:j+80] + '\n')
                
                # Write to RCL_fasta if requested
                if rcl_fasta_out:
                    rcl_fasta_out.write(f">{accession} RCL {rcl_start_1} - {rcl_end_1}\n")
                    for j in range(0, len(rcl_sequence), 80):
                        rcl_fasta_out.write(rcl_sequence[j:j+80] + '\n')
    
    # Close output files
    tsv_out.close()
    if seq_fasta_out:
        seq_fasta_out.close()
    if rcl_fasta_out:
        rcl_fasta_out.close()
    
    # Print file paths
    print(f"\n✓ RCL locations saved to: {tsv_file}")
    if seq_fasta_file:
        print(f"✓ Sequences with RCL saved to: {seq_fasta_file}")
    if rcl_fasta_file:
        print(f"✓ RCL sequences saved to: {rcl_fasta_file}")
    
    # Calculate averages
    if num_with_rcl > 0:
        avg_confidence = total_confidence / num_with_rcl
        avg_rcl_length = total_rcl_length / num_with_rcl
    else:
        avg_confidence = 0.0
        avg_rcl_length = 0.0
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total sequences processed: {len(sequences)}")
    print(f"Sequences with RCL (≥{args.min_rcl_length} aa): {num_with_rcl} ({100*num_with_rcl/len(sequences):.1f}%)")
    if num_with_rcl > 0:
        print(f"Average RCL length: {avg_rcl_length:.1f} amino acids")
        print(f"Average confidence: {avg_confidence:.3f}")
    print("="*70)
    
    print("\n✓ Prediction complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
