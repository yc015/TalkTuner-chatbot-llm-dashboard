#!/usr/bin/env python3
"""
Test script to verify the probing system setup.

This script checks:
1. All required dependencies are installed
2. The Flask server is running
3. Models are loaded properly
4. Basic functionality works
"""

import sys
import os


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sklearn", "scikit-learn"),
        ("flask", "Flask"),
        ("openai", "OpenAI"),
        ("tqdm", "tqdm"),
        ("numpy", "NumPy")
    ]
    
    missing = []
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✓ All packages installed\n")
    return True


def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  ⚠ CUDA not available - will run on CPU (very slow)")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}\n")
        return False


def test_imports_local():
    """Test that local modules can be imported."""
    print("Testing local modules...")
    
    # Add parent directory to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))
    
    try:
        from chat.classifiers import LinearProbeClassification
        print("  ✓ LinearProbeClassification imported")
    except Exception as e:
        print(f"  ✗ Failed to import LinearProbeClassification: {e}")
        return False
    
    try:
        from generate_conversation_dataset import generate_dataset
        print("  ✓ generate_dataset imported")
    except Exception as e:
        print(f"  ✗ Failed to import generate_dataset: {e}")
        return False
    
    try:
        from train_probes import train_probes_from_dataset, MODEL_CONFIGS
        print("  ✓ train_probes imported")
        print(f"  ✓ Supported models: {list(MODEL_CONFIGS.keys())}")
    except Exception as e:
        print(f"  ✗ Failed to import train_probes: {e}")
        return False
    
    print("✓ All local modules imported successfully\n")
    return True


def test_server():
    """Test if Flask server is running."""
    print("Testing Flask server...")
    try:
        import requests
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("  ✓ Server is running")
            print(f"  ✓ Status: {data.get('status')}")
            
            models = data.get('models', {})
            for model_name, model_info in models.items():
                if model_info.get('loaded'):
                    print(f"  ✓ {model_name}: loaded")
                else:
                    print(f"  ⚠ {model_name}: not loaded")
            
            print()
            return True
        else:
            print(f"  ✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ⚠ Server not running")
        print("  Start with: bash start_server.sh")
        print()
        return False
    except Exception as e:
        print(f"  ✗ Error connecting to server: {e}\n")
        return False


def test_directories():
    """Test that necessary directories exist."""
    print("Testing directories...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    required_dirs = [
        "probing_datasets",
        "gemma2_control_probes",
        "gemma2_read_probes",
        "llama3_control_probes",
        "llama3_read_probes"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ℹ {dir_name}/ - will be created when needed")
    
    print()
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Probing System Setup Test")
    print("="*60)
    print()
    
    results = {
        "Imports": test_imports(),
        "CUDA": test_cuda(),
        "Local Modules": test_imports_local(),
        "Directories": test_directories(),
        "Flask Server": test_server()
    }
    
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! System is ready.")
    else:
        print("\n⚠ Some tests failed. See details above.")
        if not results["Flask Server"]:
            print("\nNote: Flask server test will fail if server is not running.")
            print("This is expected if you haven't started the server yet.")
    
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

