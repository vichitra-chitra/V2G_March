# setup_env.py
import os
import subprocess
import sys

VENV_DIR = ".venv"

def create_virtualenv():
    if not os.path.isdir(VENV_DIR):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("Virtual environment already exists.")

def install_requirements():
    req_file = "requirements.txt"
    if not os.path.isfile(req_file):
        print("requirements.txt not found, skipping.")
        return

    pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "pip")
    print("Installing / updating packages...")
    subprocess.check_call([pip_exe, "install", "-r", req_file])

def main():
    create_virtualenv()
    install_requirements()
    print("\nSetup completed ✔")

if __name__ == "__main__":
    main()