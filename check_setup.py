#!/usr/bin/env python3
"""
Quick sanity check script to verify the RCL predictor setup.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠ Warning: Python 3.8+ recommended")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'Bio': 'Biopython',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (missing)")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
        return False
    return True

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f"✓ {gpu_count} GPU(s) detected")
            for i in range(gpu_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            
            # Multi-GPU recommendation
            if gpu_count > 1:
                print(f"\n💡 You have {gpu_count} GPUs! Use --multi-gpu flag for faster training")
                print(f"   Example: python src/train.py --encoding blosum --model unet --multi-gpu")
        else:
            print("✓ CUDA not available (CPU only)")
    except:
        print("✗ Could not check CUDA")

def check_project_structure():
    """Check if project structure is correct."""
    required_dirs = [
        'src/data',
        'src/models',
        'src/utils',
        'data/encodings'
    ]
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'src/train.py',
        'src/evaluate.py',
        'src/predict.py',
        'data/encodings/One_hot.json',
        'data/encodings/BLOSUM62.json'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).is_file():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_good = False
    
    return all_good

def check_imports():
    """Check if custom modules can be imported."""
    sys.path.insert(0, 'src')
    
    try:
        from data import get_encoder
        print("✓ data module")
    except Exception as e:
        print(f"✗ data module: {e}")
        return False
    
    try:
        from models import get_model
        print("✓ models module")
    except Exception as e:
        print(f"✗ models module: {e}")
        return False
    
    try:
        from utils import compute_metrics
        print("✓ utils module")
    except Exception as e:
        print(f"✗ utils module: {e}")
        return False
    
    return True

def check_data_access():
    """Check if training data is accessible."""
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_files = [
        config['data']['train_serpin'],
        config['data']['train_non_serpin'],
        config['data']['fasta_annotations']
    ]
    
    for data_file in data_files:
        if Path(data_file).is_file():
            print(f"✓ {data_file}")
        else:
            print(f"⚠ {data_file} (not found - check path in config.yaml)")

def test_encoding():
    """Test encoding functionality."""
    sys.path.insert(0, 'src')
    
    try:
        from data import get_encoder
        
        sequence = "MLKIVILVTFPLVCF"
        
        # Test one-hot encoding
        encoder = get_encoder('onehot', max_length=100)
        encoded = encoder.encode(sequence)
        assert encoded.shape == (100, 21), f"Expected (100, 21), got {encoded.shape}"
        print(f"✓ One-hot encoding: {encoded.shape}")
        
        # Test BLOSUM encoding
        encoder = get_encoder('blosum', max_length=100)
        encoded = encoder.encode(sequence)
        assert encoded.shape == (100, 20), f"Expected (100, 20), got {encoded.shape}"
        print(f"✓ BLOSUM encoding: {encoded.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Encoding test failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    sys.path.insert(0, 'src')
    
    try:
        import yaml
        from models import get_model
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test CNN
        model = get_model('cnn', 21, config['models']['cnn'])
        print(f"✓ CNN model: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test U-Net
        model = get_model('unet', 21, config['models']['unet'])
        print(f"✓ U-Net model: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test LSTM
        model = get_model('lstm', 21, config['models']['lstm'])
        print(f"✓ LSTM model: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def main():
    """Run all checks."""
    print("="*60)
    print("RCL Predictor - System Check")
    print("="*60)
    
    print("\n[1/8] Checking Python version...")
    check_python_version()
    
    print("\n[2/8] Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n[3/8] Checking CUDA...")
    check_cuda()
    
    print("\n[4/8] Checking project structure...")
    struct_ok = check_project_structure()
    
    if not deps_ok:
        print("\n⚠ Please install dependencies first: pip install -r requirements.txt")
        return
    
    print("\n[5/8] Checking module imports...")
    imports_ok = check_imports()
    
    print("\n[6/8] Checking data access...")
    check_data_access()
    
    print("\n[7/8] Testing encoding...")
    encoding_ok = test_encoding()
    
    print("\n[8/8] Testing model creation...")
    model_ok = test_model_creation()
    
    print("\n" + "="*60)
    if all([deps_ok, struct_ok, imports_ok, encoding_ok, model_ok]):
        print("✓ All checks passed! Ready to train models.")
        print("\nNext steps:")
        print("  1. Review config.yaml")
        print("  2. python src/train.py --encoding onehot --model cnn --epochs 5")
        print("  3. See USAGE.md for more examples")
    else:
        print("⚠ Some checks failed. Please fix issues above.")
    print("="*60)

if __name__ == '__main__':
    main()
