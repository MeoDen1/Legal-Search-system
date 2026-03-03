import os
import pandas as pd
from loguru import logger

from datetime import datetime
from .dataframe_builder import DataframeBuilder
from .dataframe_processor import DataframeProcessor
from ...utils.data_utils import get_metadata, save_metadata

class DatasetBuilder:
    def __init__(
        self,
        data_path: str
    ):
        self.data_path = data_path
        self.bronze_path = os.path.join(self.data_path, "bronze")
        self.silver_path = os.path.join(self.data_path, "silver")
        self.cluster_path = os.path.join(self.silver_path, "cluster.json")
        self.jsons_path = os.path.join(self.silver_path, "document_jsons")
        self.dataset_path = os.path.join(self.silver_path, "datasets")
        self.metadata = get_metadata(self.silver_path)


        self.output_csv = os.path.join(self.dataset_path, "dataset.csv")
        os.makedirs(os.path.join(self.silver_path, "datasets"), exist_ok=True)

        self.dataframe_builder = DataframeBuilder(self.jsons_path, self.cluster_path)
        self.dataframe_processor = DataframeProcessor()

    def _update_metadata(self):
        self.metadata["datasets"] = {}
        self.metadata["datasets"]["csv"] = self.output_csv
        self.metadata["datasets"]["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        save_metadata(self.silver_path, self.metadata)


    def __call__(self):
        # Build Dataset from JSON documents
        df = self.dataframe_builder()

        # Preprocess dataframe
        df = self.dataframe_processor(df)

        # Save to CSV
        df.to_csv(self.output_csv, index=False, encoding="utf-8")

        # Update metadata
        self._update_metadata()

        return df
