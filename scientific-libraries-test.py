#!/usr/bin/env python3
"""
Termux Scientific Libraries Test Script
======================================
This script tests if common data science libraries are properly installed and functional.
Run this after installing the data science environment to verify everything works.
"""

import os
import sys
import time
import importlib
from importlib import util

# ANSI colors for better output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

def print_header(text):
    """Print a formatted header"""
    print(f"\n{BLUE}{BOLD}{'='*60}{ENDC}")
    print(f"{BLUE}{BOLD} {text}{ENDC}")
    print(f"{BLUE}{BOLD}{'='*60}{ENDC}")

def check_import(module_name, submodules=None):
    """Test if a module can be imported and optionally test its submodules"""
    try:
        print(f"Testing {BOLD}{module_name}{ENDC}...", end="", flush=True)
        module = importlib.import_module(module_name)
        
        if hasattr(module, "__version__"):
            version = module.__version__
        elif hasattr(module, "version"):
            version = module.version
        elif hasattr(module, "VERSION"):
            version = module.VERSION
        else:
            version = "unknown"
            
        print(f" {GREEN}OK{ENDC} (version: {version})")
        
        # Test submodules if provided
        if submodules:
            for submodule in submodules:
                full_name = f"{module_name}.{submodule}"
                try:
                    importlib.import_module(full_name)
                    print(f"  └─ Submodule {submodule}: {GREEN}OK{ENDC}")
                except Exception as e:
                    print(f"  └─ Submodule {submodule}: {RED}FAILED{ENDC}")
                    print(f"     └─ Error: {str(e)}")
                    
        return True
    except Exception as e:
        print(f" {RED}FAILED{ENDC}")
        print(f"  └─ Error: {str(e)}")
        return False

def run_function_test(title, test_function):
    """Run a test function and report results"""
    print(f"Testing {BOLD}{title}{ENDC}...", end="", flush=True)
    try:
        start_time = time.time()
        test_function()
        elapsed = time.time() - start_time
        print(f" {GREEN}OK{ENDC} ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f" {RED}FAILED{ENDC}")
        print(f"  └─ Error: {str(e)}")
        return False

def test_numpy():
    """Test NumPy functionality"""
    import numpy as np
    # Create an array and perform operations
    arr = np.arange(1000).reshape(10, 100)
    result = np.mean(arr, axis=0)
    assert result.shape == (100,), "Unexpected shape"

def test_scipy():
    """Test SciPy functionality"""
    from scipy import stats
    import numpy as np
    # Generate some random data and test statistical functions
    data = np.random.normal(0, 1, 100)
    result = stats.describe(data)
    assert len(result) > 0, "Empty result"

def test_pandas():
    """Test pandas functionality"""
    import pandas as pd
    # Create a DataFrame and manipulate it
    df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
    result = df.describe()
    assert 'A' in result.columns, "Column A not found"

def test_matplotlib():
    """Test matplotlib plotting"""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a simple plot
    plt.figure(figsize=(4, 3))
    x = np.linspace(0, 10, 100)
    plt.plot(x, np.sin(x))
    plt.title("Sine Wave")
    plt.close()

def test_seaborn():
    """Test seaborn plotting"""
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create a sample dataset
    data = pd.DataFrame({
        'x': range(10),
        'y': range(10),
        'category': ['A', 'B'] * 5
    })
    
    # Create a simple plot
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='x', y='y', hue='category', data=data)
    plt.close()

def test_scikit_learn():
    """Test scikit-learn functionality"""
    from sklearn import datasets, svm
    # Load a dataset and fit a simple model
    digits = datasets.load_digits()
    X, y = digits.data[:10], digits.target[:10]
    clf = svm.SVC()
    clf.fit(X, y)

def test_statsmodels():
    """Test statsmodels functionality"""
    import statsmodels.api as sm
    import numpy as np
    # Generate data for a simple regression
    x = np.arange(10)
    X = sm.add_constant(x)
    y = x * 2 + 1 + np.random.normal(size=len(x))
    # Fit the model
    model = sm.OLS(y, X)
    results = model.fit()
    assert results.params.shape == (2,), "Unexpected shape"

def test_opencv():
    """Test OpenCV functionality"""
    import cv2
    import numpy as np
    # Create a simple image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    # Apply a blur
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    assert blurred.shape == img.shape, "Shape changed unexpectedly"

def main():
    """Main test function"""
    print_header("Termux Scientific Libraries Test")
    print(f"{YELLOW}Python version: {sys.version}{ENDC}")
    print(f"{YELLOW}Python executable: {sys.executable}{ENDC}")
    
    # List of libraries to test
    libraries = [
        ("numpy", ["random", "linalg", "fft"]),
        ("scipy", ["stats", "optimize", "integrate"]),
        ("pandas", ["core", "io"]),
        ("matplotlib", ["pyplot"]),
        ("seaborn", None),
        ("sklearn", ["datasets", "svm", "metrics"]),
        ("statsmodels", ["api"]),
        ("cv2", None),  # OpenCV
        ("jupyterlab", None),
        ("openpyxl", None),
    ]
    
    # Test imports
    print_header("Testing Library Imports")
    import_results = []
    for lib, submodules in libraries:
        import_results.append((lib, check_import(lib, submodules)))
    
    # Test functionality
    print_header("Testing Library Functionality")
    function_tests = [
        ("NumPy array operations", test_numpy),
        ("SciPy statistical functions", test_scipy),
        ("Pandas DataFrame operations", test_pandas),
        ("Matplotlib plotting", test_matplotlib),
        ("Seaborn plotting", test_seaborn),
        ("Scikit-learn model fitting", test_scikit_learn),
        ("Statsmodels regression", test_statsmodels),
        ("OpenCV image processing", test_opencv),
    ]
    
    function_results = []
    for title, func in function_tests:
        function_results.append((title, run_function_test(title, func)))
    
    # Summary
    print_header("Test Summary")
    
    print(f"{BOLD}Import Tests:{ENDC}")
    success_imports = sum(1 for _, result in import_results if result)
    print(f"Passed: {success_imports}/{len(import_results)}")
    
    if len(function_results) > 0:
        print(f"\n{BOLD}Functionality Tests:{ENDC}")
        success_functions = sum(1 for _, result in function_results if result)
        print(f"Passed: {success_functions}/{len(function_results)}")
    
    # List failed tests
    failed_imports = [(lib, i+1) for i, (lib, result) in enumerate(import_results) if not result]
    failed_functions = [(test, i+1) for i, (test, result) in enumerate(function_results) if not result]
    
    if failed_imports:
        print(f"\n{RED}Failed Import Tests:{ENDC}")
        for lib, num in failed_imports:
            print(f"  {num}. {lib}")
    
    if failed_functions:
        print(f"\n{RED}Failed Functionality Tests:{ENDC}")
        for test, num in failed_functions:
            print(f"  {num}. {test}")
    
    # Overall assessment
    total_tests = len(import_results) + len(function_results)
    total_passed = success_imports + success_functions
    
    print(f"\n{BOLD}Overall Result:{ENDC}", end=" ")
    if total_passed == total_tests:
        print(f"{GREEN}ALL TESTS PASSED{ENDC}")
    else:
        print(f"{RED}{total_passed}/{total_tests} TESTS PASSED{ENDC}")
    
    return 0 if total_passed == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())
