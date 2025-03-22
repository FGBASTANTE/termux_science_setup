# Termux Science Setup

A collection of scripts to set up a complete data science environment in Termux on Android devices.

## Overview

This repository provides scripts to easily install and test a comprehensive scientific computing and data analysis environment on Android using Termux. With these scripts, you can transform your Android device into a portable data science workstation with Jupyter notebooks and all the popular scientific Python libraries.

## Features

- **One-command installation** of a complete data science stack
- **Automatic compatibility fixes** for Android-specific issues
- **Dynamic detection** of Python version and Android API level
- **Comprehensive test suite** to verify your installation
- **Support for popular libraries** including:
  - NumPy, SciPy, Pandas
  - Matplotlib, Seaborn
  - Scikit-learn, Statsmodels
  - OpenCV
  - Jupyter Notebook

## Requirements

- Android device
- [Termux](https://github.com/termux/termux-app) (available on F-Droid)
- At least 4GB of free storage
- 4GB or more RAM recommended

## Installation

1. Install Termux from F-Droid
2. Open Termux and run the following commands:

```bash
pkg install git
git clone https://github.com/FGBASTANTE/termux_science_setup.git
cd termux_science_setup
chmod +x setup_science.sh
./setup_science.sh
```

The installation process may take 30-60 minutes depending on your device's speed and internet connection.

## Testing Your Installation

After installation, you can verify that everything is working correctly by running the test script:

```
python test_libraries.py
```

This will check that all libraries are properly installed and functioning.

## Usage

Once installation is complete, you can start a Jupyter server:

```bash
jupyter lab
```

Since Termux doesn't have a built-in browser, you'll need to:

1. Note the URL and token from the Jupyter output
2. Open a browser on your device and navigate to that URL
   - Typically `http://localhost:8888/?token=<your_token>`

Alternatively, you can access it from another device on the same network by using your device's IP address.

## Troubleshooting

### Common Issues

1. **Out of memory errors**
   - Close other apps on your device
   - Restart Termux and try again

2. **Package installation failures**
   - Run `pkg update && pkg upgrade` and try again
   - Check your internet connection

3. **Library import errors**
   - Check the test script output for details
   - Try reinstalling the specific package

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Termux developers for creating an amazing terminal emulator
- The Python scientific computing community
- All contributors to this project
