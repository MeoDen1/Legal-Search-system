import os
import yaml
import json
import threading

from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
from loguru import logger

from .parser import ParserV1
from ...utils.data_utils import get_metadata, save_metadata

# This layer involves in converting html file to json structure format
# and cluster document
class Processor:
    def __init__(
        self,
        data_path: str,
        document_tree_depth: int
    ):
        self.data_path = data_path
        self.document_tree_depth = document_tree_depth
        self.bronze_path = os.path.join(self.data_path, "bronze")
        self.silver_path = os.path.join(self.data_path, "silver")
        self.html_path = os.path.join(self.bronze_path, "document_htmls")
        self.json_path = os.path.join(self.silver_path, "document_jsons")

        os.makedirs(self.silver_path, exist_ok=True)
        os.makedirs(self.json_path, exist_ok=True)

        self.bronze_metadata: Dict[str, Any] = get_metadata(self.bronze_path)
        self.silver_metadata: Dict[str, Any] = get_metadata(self.silver_path)

        self.lock = threading.Lock()
        # ScraperAPI max concurrency
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _update_metadata(
        self,
        uid: str,
        name: str,
        save_path: str
    ):
        
        if not self.silver_metadata.get("files"):
            self.silver_metadata["files"] = {}
        self.silver_metadata["files"][uid] = {}
        self.silver_metadata["files"][uid]["name"] = name
        self.silver_metadata["files"][uid]["json_path"] = save_path
        self.silver_metadata["files"][uid]["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        save_metadata(self.silver_path, self.silver_metadata)

    def _worker(self, uid, file_metadata: Dict[str, Any], pbar):
        html_path = file_metadata["html_path"]
        name = file_metadata["name"]

        pbar.set_description(f"Processing {file_metadata['name']}")

        with open(html_path, "r", encoding="utf-8") as fp:
            page_source = fp.read()

        doc_metadata : str = json.dumps({
            "name": name,
            "link": file_metadata["link"],
            "update_time": file_metadata["update_time"],
        })

        root_node = ParserV1.parse(page_source, self.document_tree_depth)
        root_node["uid"] = uid
        root_node["key"] = file_metadata["name"]
        root_node["value"] = doc_metadata


        # Save file and update metdata
        safe_uid = uid.replace("/", "_")
        file_path = os.path.join(self.json_path, f"{safe_uid}.json")
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(root_node, fp, indent=4, ensure_ascii=False)

        # Update metadata
        self._update_metadata(uid, name, file_path)


    def __call__(self):
        # Read bronze metadata to get list of html file and parse it
        items = list(self.bronze_metadata["files"].items())
        with tqdm(total=len(items), desc="Processing") as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(self._worker, uid, website_data, pbar)
                    for uid, website_data in items
                ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error("Failed to process file: " + str(e))
                pbar.update(1)

        save_metadata(self.silver_path, self.silver_metadata)
