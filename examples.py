"""
Example script demonstrating how to use the RCL predictor programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
from data import get_encoder, read_fasta
from models import get_model
import json


def example_encoding():
    """Example: Encode a protein sequence."""
    print("="*60)
    print("Example 1: Encoding a protein sequence")
    print("="*60)
    
    # Sample serpin sequence
    sequence = "MYLKIVILVTFPLVCFTQDDTPLSKPMAIDYQAEFAWDLYKKLQLGFTQNLAIAPYSL"
    
    # One-hot encoding
    encoder = get_encoder('onehot', max_length=1024)
    encoded = encoder.encode(sequence)
    
    print(f"Original sequence: {sequence}")
    print(f"Sequence length: {len(sequence)}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoding dimension: {encoder.get_encoding_dim()}")
    print()


def example_model_creation():
    """Example: Create and inspect models."""
    print("="*60)
    print("Example 2: Creating models")
    print("="*60)
    
    input_dim = 21  # One-hot encoding dimension
    
    # Load config
    with open('config.yaml', 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Create CNN model
    cnn_model = get_model('cnn', input_dim, config['models']['cnn'])
    print(f"CNN parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")
    
    # Create U-Net model
    unet_model = get_model('unet', input_dim, config['models']['unet'])
    print(f"U-Net parameters: {sum(p.numel() for p in unet_model.parameters()):,}")
    
    # Create LSTM model
    lstm_model = get_model('lstm', input_dim, config['models']['lstm'])
    print(f"LSTM parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    print()


def example_prediction():
    """Example: Make a prediction with a trained model."""
    print("="*60)
    print("Example 3: Making predictions (requires trained model)")
    print("="*60)
    
    # Check if a trained model exists
    model_dir = Path('runs')
    run_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    if not run_dirs:
        print("No trained models found. Please train a model first:")
        print("  python src/train.py --encoding onehot --model cnn --epochs 5")
        print()
        return
    
    latest_run = run_dirs[-1]
    print(f"Using model from: {latest_run}")
    
    # Load model config
    with open(latest_run / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Sample sequence
    sequence = "MYLKIVILVTFPLVCFTQDDTPLSKPMAIDYQAEFAWDLYKKLQLGFTQNLAI"
    
    # Encode
    encoder = get_encoder(config['encoding'], max_length=config['data']['max_length'])
    encoded = encoder.encode(sequence)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoding_dims = {'onehot': 21, 'blosum': 20, 'esm2': 1280}
    model = get_model(
        config['model'],
        encoding_dims[config['encoding']],
        config['models'][config['model']]
    )
    
    checkpoint = torch.load(latest_run / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Predict
    with torch.no_grad():
        input_tensor = torch.from_numpy(encoded).unsqueeze(0).float().to(device)
        output = model(input_tensor)
        probs = torch.softmax(output, dim=-1)
    
    # Get RCL probabilities
    rcl_probs = probs[0, :len(sequence), 1].cpu().numpy()
    
    print(f"Sequence: {sequence}")
    print(f"RCL probabilities (first 10 positions): {rcl_probs[:10]}")
    
    # Find predicted RCL region
    threshold = 0.5
    rcl_positions = np.where(rcl_probs > threshold)[0]
    
    if len(rcl_positions) > 0:
        print(f"Predicted RCL positions: {rcl_positions.tolist()}")
        print(f"Predicted RCL sequence: {sequence[rcl_positions[0]:rcl_positions[-1]+1]}")
    else:
        print("No RCL region detected above threshold")
    print()


def example_data_loading():
    """Example: Load and inspect training data."""
    print("="*60)
    print("Example 4: Loading training data")
    print("="*60)
    
    from data import read_csv_with_annotations
    
    # Load a sample CSV
    csv_file = "../rcl-unet/data/Alphafold_RCL_annotations.csv"
    
    if not Path(csv_file).exists():
        print(f"Sample data not found: {csv_file}")
        print()
        return
    
    ids, labels, sequences, rcl_seqs = read_csv_with_annotations(csv_file, max_length=1024)
    
    print(f"Loaded {len(ids)} sequences")
    print(f"First protein ID: {ids[0]}")
    print(f"First sequence length: {len(sequences[0])}")
    print(f"Label shape: {labels[0].shape}")
    
    # Count RCL positions in first sequence
    rcl_count = np.sum(labels[0][:, 1] == 1)
    print(f"RCL residues in first sequence: {rcl_count}")
    
    if rcl_seqs[0]:
        print(f"RCL sequence: {rcl_seqs[0]}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("RCL Predictor - Usage Examples")
    print("*" * 60)
    print()
    
    try:
        example_encoding()
    except Exception as e:
        print(f"Error in encoding example: {e}\n")
    
    try:
        example_model_creation()
    except Exception as e:
        print(f"Error in model creation example: {e}\n")
    
    try:
        example_data_loading()
    except Exception as e:
        print(f"Error in data loading example: {e}\n")
    
    try:
        example_prediction()
    except Exception as e:
        print(f"Error in prediction example: {e}\n")
    
    print("*" * 60)
    print("Examples complete!")
    print("*" * 60)
    print()


if __name__ == '__main__':
    main()
