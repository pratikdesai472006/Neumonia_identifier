"""Automated dependency verification and installer for PneumoVision.

Checks all required scientific, machine learning, and UI dependencies.
If any required library is missing, installs it automatically before running.
"""

import importlib
import importlib.util
import subprocess
import sys
from typing import List, Dict, Tuple

# Mapping of Python import name to PyPI package specification
REQUIRED_PACKAGES: List[Dict[str, str]] = [
    {"import_name": "torch", "pip_name": "torch>=2.0.0", "description": "PyTorch Deep Learning Engine"},
    {"import_name": "torchvision", "pip_name": "torchvision>=0.15.0", "description": "Vision Models & Transforms"},
    {"import_name": "pydicom", "pip_name": "pydicom>=2.4.0", "description": "Medical DICOM Radiograph Parser"},
    {"import_name": "cv2", "pip_name": "opencv-python>=4.8.0", "description": "OpenCV Computer Vision"},
    {"import_name": "PIL", "pip_name": "pillow>=9.5.0", "description": "Pillow Image Processing"},
    {"import_name": "numpy", "pip_name": "numpy>=1.24.0", "description": "NumPy Numerical Computing"},
    {"import_name": "pandas", "pip_name": "pandas>=2.0.0", "description": "Pandas Data Analysis"},
    {"import_name": "sklearn", "pip_name": "scikit-learn>=1.3.0", "description": "Scikit-Learn Evaluation Metrics"},
    {"import_name": "skimage", "pip_name": "scikit-image>=0.20.0", "description": "Scikit-Image Filters"},
    {"import_name": "matplotlib", "pip_name": "matplotlib>=3.7.0", "description": "Matplotlib Charting & ROC Plots"},
    {"import_name": "tqdm", "pip_name": "tqdm>=4.65.0", "description": "Progress Bar Telemetry"},
    {"import_name": "kaggle", "pip_name": "kaggle>=1.5.0", "description": "Kaggle Dataset API"},
    {"import_name": "kagglehub", "pip_name": "kagglehub>=0.2.0", "description": "Kaggle Hub Downloader"},
    {"import_name": "streamlit", "pip_name": "streamlit>=1.25.0", "description": "Streamlit Web Portal UI"},
    {"import_name": "reportlab", "pip_name": "reportlab>=4.0.0", "description": "PDF Clinical Report Generator"},
    {"import_name": "docx", "pip_name": "python-docx>=1.1.0", "description": "Word DOCX Report Generator"},
]


def is_installed(import_name: str) -> bool:
    """Check if a module is installed and importable without executing heavy C-extension init."""
    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except (ImportError, ValueError, AttributeError):
        try:
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False


def install_package(pip_name: str) -> bool:
    """Install a package via pip using the currently executing Python interpreter."""
    print(f"[PneumoVision] Installing missing package: '{pip_name}' ...", flush=True)
    cmd = [sys.executable, "-m", "pip", "install", pip_name]
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except Exception as exc:
        print(f"[PneumoVision] [ERROR] Failed to install {pip_name}: {exc}", flush=True)
        return False


def check_and_install_dependencies(
    auto_install: bool = True,
    verbose: bool = True,
    packages: List[Dict[str, str]] = None
) -> Tuple[bool, List[str]]:
    """Check that all required dependencies are installed.
    
    If auto_install is True, missing packages are installed automatically.
    Returns (success_boolean, list_of_missing_packages).
    """
    pkg_list = packages or REQUIRED_PACKAGES
    missing = []
    
    if verbose:
        print("=" * 65, flush=True)
        print("PneumoVision Dependency Verification & Auto-Installer", flush=True)
        print("=" * 65, flush=True)
    
    for item in pkg_list:
        imp = item["import_name"]
        pip_pkg = item["pip_name"]
        desc = item.get("description", imp)
        
        if is_installed(imp):
            if verbose:
                print(f"  [OK]      {imp:<14} -> {desc}", flush=True)
        else:
            if verbose:
                print(f"  [MISSING] {imp:<14} ({pip_pkg})", flush=True)
            if auto_install:
                success = install_package(pip_pkg)
                if success and is_installed(imp):
                    if verbose:
                        print(f"            Successfully installed {imp}!", flush=True)
                else:
                    missing.append(pip_pkg)
            else:
                missing.append(pip_pkg)
                
    if verbose:
        print("-" * 65, flush=True)
        if not missing:
            print("All required libraries are verified and ready to run!", flush=True)
        else:
            print(f"Warning: Could not resolve packages: {', '.join(missing)}", flush=True)
            print(f"Run manually in terminal: pip install -r requirements.txt", flush=True)
        print("=" * 65, flush=True)
        
    return len(missing) == 0, missing


if __name__ == "__main__":
    check_and_install_dependencies(auto_install=True, verbose=True)
