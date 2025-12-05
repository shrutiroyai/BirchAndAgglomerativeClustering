import os
import sys
import subprocess
import importlib.util

def install_requirements(requirements_file='requirements.txt'):
    """
    Checks for a requirements.txt file and installs the packages listed within it.
    Exits the script if the requirements file is not found.
    """
    if not os.path.exists(requirements_file):
        print(f"Error: '{requirements_file}' not found. Cannot check/install dependencies.")
        sys.exit(1)

    # Check if pip is available for the current Python interpreter
    pip_spec = importlib.util.find_spec("pip")
    if pip_spec is None:
        print("Error: 'pip' is not installed for the current Python interpreter.")
        print(f"Please install pip for: {sys.executable}")
        sys.exit(1)

    print("Checking and installing required packages...")
    try:
        # Use sys.executable to ensure pip from the correct Python environment is used.
        # This is more robust than calling 'pip' or 'pip3' directly.
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-r', requirements_file]
        )
        print("...installation check complete.")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to install packages. The 'pip' command failed with exit code {e.returncode}.")
        print(f"Please try running 'pip install -r {requirements_file}' manually to see the full error details.")
        sys.exit(1)