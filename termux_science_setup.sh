#!/bin/bash
# Install Jupyter lab and data science packages on Termux
# Exit on error
set -e

echo "==== Setting up Data Science Environment in Termux ===="

# Function to check if a command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed, exiting"
        exit 1
    fi
}

# Update and upgrade packages
echo "Updating Termux packages..."
pkg update -y && pkg upgrade -y
check_success "Package update"

# Install core dependencies
echo "Installing core dependencies..."
pkg install -y git clang python libzmq rust binutils cmake wget which patchelf
check_success "Core dependencies installation"

# Install build tools first (needed for many packages)
echo "Installing build tools..."
pkg install -y build-essential ninja flang libopenblas libandroid-execinfo patchelf binutils-is-llvm
check_success "Build tools installation"

# Install Python packaging tools
echo "Installing Python packaging tools..."
pip install maturin setuptools wheel packaging pyproject_metadata cython meson-python versioneer setuptools-scm pythran patsy formulaic pytest
check_success "Python packaging tools installation"

# Get Python version dynamically
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python $PYTHON_VERSION"

# Fix OpenMP issues
echo "Fixing OpenMP configuration..."
_file="$(find $PREFIX/lib/python$PYTHON_VERSION -name "*sysconfigdata*.py")"
if [ -z "$_file" ]; then
    echo "Error: Could not find sysconfigdata file"
    exit 1
fi

rm -rf $PREFIX/lib/python$PYTHON_VERSION/__pycache__
cp "$_file" "$_file.backup"
sed -i 's|-fno-openmp-implicit-rpath||g' "$_file"
check_success "OpenMP configuration fix"

# Install ZMQ and Jupyter
echo "Installing pyzmq and Jupyter..."
pip install pyzmq
pip install jupyter
check_success "Jupyter installation"

# Patch ZMQ library with correct Python version
echo "Patching ZMQ library..."
ZMQ_SO_PATH="/data/data/com.termux/files/usr/lib/python$PYTHON_VERSION/site-packages/zmq/backend/cython/_zmq.cpython-${PYTHON_VERSION//./}.so"
if [ -f "$ZMQ_SO_PATH" ]; then
    patchelf --add-needed libpython$PYTHON_VERSION.so "$ZMQ_SO_PATH"
else
    echo "Warning: ZMQ library not found at expected location."
    # Try to find it
    ZMQ_SO_PATH=$(find $PREFIX/lib/python$PYTHON_VERSION -name "_zmq.cpython-*.so")
    if [ -n "$ZMQ_SO_PATH" ]; then
        patchelf --add-needed libpython$PYTHON_VERSION.so "$ZMQ_SO_PATH"
    else
        echo "Error: Could not find ZMQ library to patch."
        exit 1
    fi
fi
check_success "ZMQ patching"

# Install scientific libraries
echo "Installing scientific packages (this may take some time)..."
pkg install -y matplotlib python-numpy python-scipy python-pyarrow
check_success "Scientific packages installation"

# Install pandas with correct flags
echo "Installing pandas..."
LDFLAGS="-lpython$PYTHON_VERSION" pip3 install --no-build-isolation --no-cache-dir pandas
check_success "Pandas installation"

# Install scikit-learn
echo "Installing scikit-learn..."
pip install scikit-learn -v --no-build-isolation
check_success "Scikit-learn installation"

# Install additional data science libraries
echo "Installing additional libraries..."
pip install seaborn openpyxl
check_success "Additional libraries installation"

# Get Android API level
API_LEVEL=$(getprop ro.build.version.sdk)
echo "Android API level: $API_LEVEL"

# Force reinstall numpy for compatibility
echo "Reinstalling numpy for better compatibility..."
pip3 install --upgrade --force-reinstall numpy
check_success "Numpy reinstallation"

# Install OpenCV
echo "Installing OpenCV..."
pkg install -y x11-repo
pkg update
pkg install -y opencv-python
check_success "OpenCV installation"

# Install statsmodels from source with proper flags
echo "Installing statsmodels from source..."
if [ ! -d "statsmodels" ]; then
    git clone https://github.com/statsmodels/statsmodels.git
    check_success "Statsmodels repository cloning"
fi
cd statsmodels
# Use detected API level instead of hardcoded 34
CFLAGS+=" -U__ANDROID_API__ -D__ANDROID_API__=$API_LEVEL" MATHLIB=m LDFLAGS="-lpython$PYTHON_VERSION" python -m pip install . --no-build-isolation
check_success "Statsmodels installation"

# Install Gemini-Cli
pkg install nodejs
npm install -g @google/gemini-cli
check_success "Gemini-Cli installation"

# Clean up
echo "Cleaning package cache..."
pip cache purge
check_success "Cache cleanup"

echo "==== Installation Complete ===="
echo "You can now run Jupyter with: jupyter lab or run basic tests with: python scientific-libraries-test.py"
