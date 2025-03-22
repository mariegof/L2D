import os
import wandb

def test_wandb_connection():
    """Test that WandB is properly installed and can connect to servers"""
    print("Testing WandB connection...")
    try:
        wandb.login()
        print("✅ WandB login successful")
        return True
    except Exception as e:
        print(f"❌ WandB login failed: {e}")
        return False

def test_directory_structure():
    """Test that the required directories exist or can be created"""
    dirs_to_check = [
        'configs',
        'data/results',
        'models'
    ]
    
    print("Testing directory structure...")
    for dir_path in dirs_to_check:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Directory {dir_path} is available")
        except Exception as e:
            print(f"❌ Cannot create directory {dir_path}: {e}")
            return False
    return True

def test_pytorch():
    """Test that PyTorch is installed and working"""
    print("Testing PyTorch installation...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} is installed")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_numpy():
    """Test that NumPy is installed and working"""
    print("Testing NumPy installation...")
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} is installed")
        # Try a simple array operation
        arr = np.random.rand(3,3)
        result = np.matmul(arr, arr)
        print("   NumPy array operations working correctly")
        return True
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("🔍 RUNNING SERVER ENVIRONMENT TESTS 🔍")
    print("=====================================")
    
    results = {
        "WandB Connection": test_wandb_connection(),
        "Directory Structure": test_directory_structure(),
        "PyTorch": test_pytorch(),
        "NumPy": test_numpy()
    }
    
    print("\n📋 TEST SUMMARY 📋")
    print("=================")
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 All tests passed! Your environment is ready.")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before proceeding.")

if __name__ == "__main__":
    run_all_tests()