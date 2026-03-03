import time
import json
import kaggle
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


def _sync_dataset_metadata(path: Path, slug: str, name: str):
    """Ensures the dataset-metadata.json is correct before CLI execution."""
    metadata = {
        "title": name,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(path / "dataset-metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Kaggle | Metadata synced for: {slug}")

def upload_dataset_to_kaggle(
    dataset_dir: str,
    dataset_name: str,
    username: str,
    reset: bool = False
):
    """
    Uploads a dataset to Kaggle using the CLI.
    Preserves folder structures (e.g., models/, vectors/) via --dir-mode zip.
    """
    path = Path(dataset_dir)
    slug = f"{username}/{dataset_name}"
    
    try:
        # 0. Handle Reset (Delete existing dataset)
        if reset:
            logger.warning(f"Kaggle | Attempting to delete dataset: {slug}")
            # subprocess.run is safer than the raw API call for deletions
            subprocess.run(["kaggle", "datasets", "delete", "-f", slug], check=False)
            logger.info("Kaggle | Delete command sent. Waiting for Kaggle backend (soft-delete)...")
            time.sleep(5) 

        # 1. Prepare Metadata
        _sync_dataset_metadata(path, slug, dataset_name)

        # 2. Check existence via CLI status
        status_check = subprocess.run(
            ["kaggle", "datasets", "status", slug],
            capture_output=True, text=True
        )
        
        # 3. Choose Create or Version command
        if "ready" in status_check.stdout.lower() and not reset:
            logger.info(f"Kaggle | Dataset exists. Pushing NEW VERSION...")
            cmd = [
                "kaggle", "datasets", "version",
                "-p", str(path),
                "-m", f"Data update: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "--dir-mode", "zip"
            ]
        else:
            logger.info(f"Kaggle | Creating NEW dataset: {slug}")
            cmd = [
                "kaggle", "datasets", "create",
                "-p", str(path),
                "--dir-mode", "zip"
            ]

        # 4. Execute CLI
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode == 0:
            logger.success(f"Kaggle | Dataset operation successful for {slug}")
        else:
            # Handle the case where 'create' fails because it exists but isn't 'Ready'
            if "already in use" in process.stderr:
                logger.warning("Kaggle | Slug in use but not ready. Retrying as version...")
                subprocess.run(["kaggle", "datasets", "version", "-p", str(path), "-m", "Retry update", "--dir-mode", "zip"])
            else:
                logger.error(f"Kaggle | CLI Error: {process.stderr}")

    except Exception as e:
        logger.error(f"Kaggle | Dataset Push Error: {e}")
