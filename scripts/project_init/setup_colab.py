"""
This script handles the initial setup for the AML Mistake Detection project in a Google Colab environment.

It automates mounting Google Drive and unzipping the required checkpoint files.

Instructions for use in a Colab Notebook:
1. Make sure you have:
   a) Your project folder ('aml-procedural-mistake-detection') in the root of your Google Drive.
   b) The 'checkpoints.zip' file located in 'My Drive/AML/CaptainCook4D/'.
2. Run the following code in a Colab cell to execute this script:

   !python "/content/drive/My Drive/aml-procedural-mistake-detection/scripts/project_init/setup_colab.py"

3. Follow the printed instructions to complete the setup.
"""

import os
import zipfile
from google.colab import drive

# --- Configuration ---
# The name of your project folder in Google Drive
PROJECT_DIR_NAME = "aml-procedural-mistake-detection"
# Path to the folder containing your zip files
ZIP_SOURCE_DIR = "/content/drive/My Drive/AML/CaptainCook4D"
# ---------------------

# The absolute path to the project directory in the Colab environment
PROJECT_PATH = f"/content/drive/My Drive/{PROJECT_DIR_NAME}"


def mount_google_drive():
    """Mounts the user's Google Drive to the '/content/drive' directory."""
    print("Mounting Google Drive...")
    if not os.path.exists('/content/drive/My Drive'):
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
    else:
        print("Google Drive is already mounted.")


def create_project_directories():
    """Creates the necessary data and checkpoint directories if they don't exist."""
    print("\nChecking for necessary directories...")

    # Directory for model checkpoints
    checkpoints_dir = os.path.join(PROJECT_PATH, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        print(f"Created directory: {checkpoints_dir}")
    else:
        print(f"Directory already exists: {checkpoints_dir}")

    # Directory for dataset features
    features_dir = os.path.join(PROJECT_PATH, "data", "features")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
        print(f"Created directory: {features_dir}")
    else:
        print(f"Directory already exists: {features_dir}")
    
    return checkpoints_dir


def unzip_files(checkpoints_dir):
    """Unzips the required files from Google Drive to their respective folders."""
    print("\nAttempting to unzip required files...")

    # Unzip checkpoints
    checkpoints_zip_path = os.path.join(ZIP_SOURCE_DIR, "checkpoints.zip")
    print(f"Looking for checkpoint zip at: {checkpoints_zip_path}")

    if os.path.exists(checkpoints_zip_path):
        try:
            with zipfile.ZipFile(checkpoints_zip_path, 'r') as zip_ref:
                zip_ref.extractall(checkpoints_dir)
            print(f"Successfully unzipped '{checkpoints_zip_path}' to '{checkpoints_dir}'")
        except Exception as e:
            print(f"ERROR: Failed to unzip {checkpoints_zip_path}. Reason: {e}")
    else:
        print("ERROR: 'checkpoints.zip' not found at the specified location.")
        print("Please ensure 'checkpoints.zip' is in 'My Drive/AML/CaptainCook4D/'")


def display_next_steps():
    """Prints the manual next steps for the user to follow in their Colab notebook."""
    print("\n--- ✅ Initial Setup Complete ---")
    print("\n--- Next Steps ---")
    print("Please run the following commands in separate Colab cells:")
    print("\n# 1. Change directory to the project folder:")
    print(f'%cd "{PROJECT_PATH}"')
    print("\n# 2. Install Python dependencies:")
    print("!pip install -r requirements.txt")
    print("\n# 3. Manually download and place the dataset:")
    print("  - The 'checkpoints.zip' file has been automatically unzipped for you.")
    print("  - ⚠️ ACTION REQUIRED: You still need to find and place the dataset features.")
    print("  - Find the link for the 'pre-extracted features', download them, and place them in:")
    print(f"  - {os.path.join(PROJECT_PATH, 'data/features/')}")
    print("\n# 4. Once the dataset is in place, you can run the evaluation script, for example:")
    print("!python -m core.evaluate --variant MLP --backbone omnivore --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt --split step --threshold 0.6")


if __name__ == "__main__":
    mount_google_drive()
    checkpoints_destination = create_project_directories()
    unzip_files(checkpoints_destination)
    display_next_steps()
