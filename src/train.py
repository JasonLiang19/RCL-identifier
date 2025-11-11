"""
Training script for RCL prediction models.
"""

import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data import (
    read_csv_with_annotations,
    parse_fasta_with_rcl_annotations,
    get_encoder,
    EncodedDataset
)
from models import get_model
from utils import (
    compute_metrics,
    MaskedBCELoss,
    MCCLoss,
    plot_training_curves
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train RCL prediction model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--encoding', type=str, choices=['onehot', 'blosum', 'esm2', 'esm2_650m'],
                       help='Encoding type (overrides config)')
    parser.add_argument('--model', type=str, choices=['cnn', 'unet', 'lstm'],
                       help='Model type (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device (overrides config)')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use all available GPUs with DataParallel')
    parser.add_argument('--gpu-ids', type=str,
                       help='Comma-separated GPU IDs to use (e.g., "0,1")')
    parser.add_argument('--output-dir', type=str, default='runs',
                       help='Output directory for runs')
    parser.add_argument('--precomputed', type=str,
                       help='Path to precomputed embeddings file (.npz)')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_next_run_dir(base_dir: str) -> Path:
    """Get the next run directory."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    existing_runs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    next_run_id = len(existing_runs) + 1
    
    run_dir = base_path / f'run_{next_run_id:03d}'
    run_dir.mkdir(parents=True)
    
    return run_dir


def load_precomputed_embeddings(embeddings_path: str):
    """Load precomputed embeddings from file using memory mapping to avoid loading all into RAM."""
    print(f"Loading precomputed embeddings from: {embeddings_path}")
    
    # Use mmap_mode='r' to avoid loading entire array into memory
    data = np.load(embeddings_path, allow_pickle=True, mmap_mode='r')
    
    # These will be memory-mapped (not loaded into RAM)
    encodings = data['encodings']
    labels = data['labels']
    
    # These are small, so load them normally
    ids = data['ids'].tolist()
    sequences = data['sequences'].tolist()
    
    print(f"Loaded {len(encodings)} sequences (memory-mapped)")
    print(f"Encoding shape: {encodings.shape}")
    print(f"Memory-mapped arrays will be loaded on-demand during training")
    
    # Get encoding dimension from the embeddings
    encoding_dim = encodings.shape[-1]
    
    return ids, encodings, labels, sequences, encoding_dim


def load_and_encode_data(config: dict, encoding_type: str, device: str):
    """Load and encode training data."""
    print("Loading data...")
    
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
    
    # Encode sequences
    print(f"\nEncoding sequences with {encoding_type}...")
    encoder = get_encoder(
        encoding_type,
        max_length=config['data']['max_length'],
        device=device
    )
    
    if encoding_type in ['esm2', 'esm2_650m']:
        batch_size = config['encoding'].get(encoding_type, {}).get('batch_size', 4)
        encodings = encoder.encode_batch(
            all_seqs,
            batch_size=batch_size
        )
    else:
        encodings = encoder.encode_batch(all_seqs)
    
    print(f"Encoding shape: {encodings.shape}")
    
    return all_ids, encodings, all_labels, all_seqs, encoder.get_encoding_dim()


def create_data_loaders(
    encodings: np.ndarray,
    labels: np.ndarray,
    ids: list,
    sequences: list,
    config: dict,
    use_memory_map: bool = False
):
    """Create train/val data loaders.
    
    Args:
        use_memory_map: If True, uses lazy loading to avoid loading memory-mapped arrays into RAM
    """
    # Split data - just get indices, don't subset arrays yet
    train_idx, val_idx = train_test_split(
        np.arange(len(encodings)),
        test_size=1 - config['data']['train_split'],
        random_state=config['data']['random_seed']
    )
    
    # Create datasets
    # For memory-mapped arrays, pass full arrays with indices for lazy loading
    # This avoids loading subsets into memory
    train_dataset = EncodedDataset(
        encodings,  # Full array
        labels,     # Full array
        ids,        # Full list
        sequences,  # Full list
        indices=train_idx,  # Subset indices
        lazy_load=use_memory_map  # Lazy tensor conversion for memory-mapped arrays
    )
    
    val_dataset = EncodedDataset(
        encodings,  # Full array
        labels,     # Full array
        ids,        # Full list
        sequences,  # Full list
        indices=val_idx,  # Subset indices
        lazy_load=use_memory_map  # Lazy tensor conversion for memory-mapped arrays
    )
    
    # Memory-mapped arrays cannot be pickled, so disable multiprocessing
    num_workers = 0 if use_memory_map else config['hardware']['num_workers']
    if use_memory_map and config['hardware']['num_workers'] > 0:
        print("⚠ Setting num_workers=0 for memory-mapped arrays (avoid pickling issues)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['hardware']['pin_memory']
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        encodings = batch['encoding'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(encodings)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            encodings = batch['encoding'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(encodings)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Collect predictions
            probs = torch.softmax(outputs, dim=-1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    encoding_type = args.encoding if args.encoding else 'onehot'
    model_type = args.model if args.model else 'cnn'
    epochs = args.epochs if args.epochs else config['training']['epochs']
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    device = args.device if args.device else config['hardware']['device']
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Report GPU info
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
    
    # Determine multi-GPU usage
    use_multi_gpu = args.multi_gpu or (config['hardware'].get('use_multi_gpu', False) and not args.gpu_ids and not args.multi_gpu)
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    elif config['hardware'].get('gpu_ids') and use_multi_gpu:
        gpu_ids = config['hardware']['gpu_ids']
    
    # Create run directory
    run_dir = get_next_run_dir(args.output_dir)
    print(f"Run directory: {run_dir}")
    
    # Save configuration
    run_config = {
        'encoding': encoding_type,
        'model': model_type,
        'epochs': epochs,
        'precomputed': args.precomputed if args.precomputed else None,
        **config
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Load data - either precomputed embeddings or encode on-the-fly
    if args.precomputed:
        ids, encodings, labels, sequences, encoding_dim = load_precomputed_embeddings(
            args.precomputed
        )
        use_memory_map = True  # Precomputed embeddings use memory mapping
    else:
        ids, encodings, labels, sequences, encoding_dim = load_and_encode_data(
            config, encoding_type, device
        )
        use_memory_map = False
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        encodings, labels, ids, sequences, config, use_memory_map=use_memory_map
    )
    
    # Create model
    print(f"\nCreating {model_type} model with {encoding_type} encoding (dim={encoding_dim})...")
    model = get_model(
        model_type,
        encoding_dim,
        config['models'][model_type],
        num_classes=2
    )
    
    # Multi-GPU setup
    if use_multi_gpu or gpu_ids:
        if torch.cuda.device_count() > 1:
            if gpu_ids:
                # Use specific GPUs
                print(f"\n🚀 Using GPUs: {gpu_ids}")
                model = nn.DataParallel(model, device_ids=gpu_ids)
                # Increase batch size for multi-GPU
                effective_batch_size = config['training']['batch_size'] * len(gpu_ids)
                print(f"Effective batch size: {effective_batch_size} ({config['training']['batch_size']} x {len(gpu_ids)} GPUs)")
            else:
                # Use all available GPUs
                print(f"\n🚀 Using ALL {torch.cuda.device_count()} GPUs (DataParallel)")
                model = nn.DataParallel(model)
                effective_batch_size = config['training']['batch_size'] * torch.cuda.device_count()
                print(f"Effective batch size: {effective_batch_size} ({config['training']['batch_size']} x {torch.cuda.device_count()} GPUs)")
        else:
            print("\n⚠ Multi-GPU requested but only 1 GPU available")
    else:
        print(f"\nUsing single GPU/CPU")
    
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    if config['training'].get('loss_weights'):
        loss_weights = torch.tensor(config['training']['loss_weights']).to(device)
    else:
        loss_weights = None
    
    criterion = MaskedBCELoss(weight=loss_weights)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=config['training']['patience'] // 2
    )
    
    # Setup tensorboard
    if config['logging']['use_tensorboard']:
        writer = SummaryWriter(run_dir / 'tensorboard')
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_metrics_history = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'mcc', 'auc']}
    
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        for metric, value in val_metrics.items():
            val_metrics_history[metric].append(value)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
        
        # Log to tensorboard
        if config['logging']['use_tensorboard']:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            for metric, value in val_metrics.items():
                writer.add_scalar(f'Metrics/{metric}', value, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            # Save model (handle DataParallel wrapper)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_f1,
                'config': run_config
            }, run_dir / 'best_model.pt')
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
    
    # Save final model
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'config': run_config
    }, run_dir / 'final_model.pt')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': {k: v for k, v in val_metrics_history.items()}
    }
    with open(run_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(
        train_losses,
        val_losses,
        val_metrics_history,
        str(run_dir)
    )
    
    if config['logging']['use_tensorboard']:
        writer.close()
    
    print(f"\n✓ Training complete!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Results saved to: {run_dir}")


if __name__ == '__main__':
    main()
