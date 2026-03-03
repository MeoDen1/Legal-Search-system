import os
import time
import json
import subprocess
import tempfile
import shutil
from loguru import logger

# ignore patterns for source code
SRC_IGNORE_PATTERNS = [
    "**/__pycache__/",
    "*.py[cod]",
    ".ipynb_checkpoints/",
    "data/",
    "model_hub/",
    ".git/",
    ".env",
    "*.pt",
    "*.csv"
]

def upload_to_kaggle(path, name, username, ignore_patterns=None):
    """Upload a file or directory to Kaggle as a dataset."""

    ignore_patterns = ignore_patterns or []
    slug = f"{username}/{name}"

    # Build the dataset metadata
    metadata = {
        "title": name,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}]
    }

    # Add dataset-metadata.json to the pushed object
    with open(os.path.join(path, "dataset-metadata.json"), "w") as fp:
        json.dump(metadata, fp, indent=4)

    # Add .kaggleignore to the pushed object
    if len(ignore_patterns) != 0:
        with open(os.path.join(path, ".kaggleignore"), "w") as fp:
            fp.write("\n".join(ignore_patterns))

    try:
        check_cmd = ["kaggle", "datasets", "status", slug]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
    
        if "ready" in result.stdout.lower():
            logger.info(f"Kaggle | Object exists. Pushing new version...")
            cmd = [
                "kaggle", "datasets", "version",
                "-p", str(path),
                "-m", f"Update {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "--dir-mode", "zip"
            ]
        else:
            logger.info(f"Kaggle | Object not found. Creating new...")
            cmd = [
                "kaggle", "datasets", "create",
                "-p", str(path),
                "--dir-mode", "zip"
            ]

        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            logger.success(f"Kaggle | Successfully pushed to {slug}")
        else:
            logger.error(f"Kaggle | CLI Error: {process.stderr}")

    except Exception as e:
        logger.error(f"Kaggle | Unexpected Error: {e}")
    
    return f"https://www.kaggle.com/datasets/{username}/{name}"
