import os
import time
import json
import subprocess
from loguru import logger
from pathlib import Path

def get_metadata(data_path: str):
    """Get metadata from json file, if not exists, create & return default metadata"""
    json_path = os.path.join(data_path, "metadata.json")
    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump({ "path": json_path }, fp, ensure_ascii=False)

    with open(json_path, "r", encoding="utf-8") as fp:
        return json.load(fp)

def save_metadata(data_path: str, metadata: dict):
    json_path = os.path.join(data_path, "metadata.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=4, ensure_ascii=False)

def get_max_value_length(json_path: str):
    max_value_length = 0
    text = ""
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

        queue = [data]
        while queue:
            item = queue.pop(0)
            if len(item["value"]) > max_value_length:
                max_value_length = len(item["value"])
                text = item["value"]
            queue.extend(item["subitems"])
    return max_value_length, text

