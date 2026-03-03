import time
import json
import subprocess

from loguru import logger
from pathlib import Path

def _prepare_kaggle_configs(path: Path, slug: str, name: str):
    """Internal helper to sync metadata and ignore files."""
    # 1. Sync dataset-metadata.json
    metadata = {
        "title": name,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(path / "dataset-metadata.json", 'w') as fp:
        json.dump(metadata, fp, indent=4)

    # 2. Sync .kaggleignore
    ignore_patterns = [
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
    with open(path / ".kaggleignore", "w") as fp:
        fp.write("\n".join(ignore_patterns))

def upload_to_kaggle(folder_path: str, name: str, username: str):
    """
    Upload source code to Kaggle using CLI to preserve directory structure.
    """
    path = Path(folder_path)
    slug = f"{username}/{name}"
    
    try:
        # Step 1: Prep files
        _prepare_kaggle_configs(path, slug, name)
        logger.info(f"Kaggle | Configs synced for {slug}")

        # Step 2: Check if dataset exists using CLI
        check_cmd = ["kaggle", "datasets", "status", slug]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        # Step 3: Execute Create or Update
        if "ready" in result.stdout.lower():
            logger.info(f"Kaggle | Object exists. Pushing new version...")
            # Using version command with --dir-mode zip
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

        # Step 4: Run CLI command
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            logger.success(f"Kaggle | Successfully pushed to {slug}")
        else:
            logger.error(f"Kaggle | CLI Error: {process.stderr}")

    except Exception as e:
        logger.error(f"Kaggle | Unexpected Error: {e}")
