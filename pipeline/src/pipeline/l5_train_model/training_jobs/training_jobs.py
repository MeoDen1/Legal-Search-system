import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from typing import Tuple, Dict, List
from loguru import logger
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..models import DecoderDataset
from .embedding import PretrainedEmbedding


class TrainingJobs:
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1):
        self.test_size = test_size
        self.val_size = val_size
        self.ros = RandomOverSampler(random_state=42)
        self.embedding = PretrainedEmbedding()
        
    def _balance_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensures all classes in the training set have an equal number of samples.
        Example: If 'Article 1' has 10 samples and 'Article 2' has 2,
        'Article 2' will be oversampled to 10.
        """
        if len(np.unique(y)) < 2:
            return X, y

        X_resampled, y_resampled = self.ros.fit_resample(X, y)
        return X_resampled, y_resampled

    def _split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Handles the 3-way stratified split and balancing."""
        # 1. Train+Val / Test split
        X_tv, X_test, y_tv, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y if len(np.unique(y)) > 1 else None,
            random_state=42,
        )

        # 2. Train / Val split
        adj_val = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv,
            y_tv,
            test_size=adj_val,
            stratify=y_tv if len(np.unique(y_tv)) > 1 else None,
            random_state=42,
        )

        # 3. Balance only the Training set
        X_train_bal, y_train_bal = self._balance_data(X_train, y_train)

        return {
            "train": (X_train_bal, y_train_bal),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def _process_layer(self, df: pd.DataFrame, depth: int) -> Dict[str, Dict]:
        """Processes all decoders for a specific layer."""
        layer_jobs = {}
        parent_col = f"label{depth}"
        child_col = f"label{depth+1}"

        if child_col not in df.columns:
            logger.warning(f"Target column {child_col} not found.")
            return {}

        unique_parents = df[parent_col].unique()

        for parent_uid in unique_parents:
            rows = df[df[parent_col] == parent_uid].copy()

            if len(rows) < 3:
                logger.warning(f"Too few samples for {parent_uid}, skipping.")
                continue

            # Matrix prep
            inputs = np.vstack(rows["input_vector"].values)
            le = LabelEncoder()
            targets = le.fit_transform(rows[child_col])

            # Split and Balance
            splits = self._split_data(inputs, targets)

            layer_jobs[parent_uid] = {
                "train_ds": DecoderDataset(
                    torch.from_numpy(splits["train"][0]).float(),
                    torch.from_numpy(splits["train"][1]).long(),
                ),
                "val_ds": DecoderDataset(
                    torch.from_numpy(splits["val"][0]).float(),
                    torch.from_numpy(splits["val"][1]).long(),
                ),
                "test_ds": DecoderDataset(
                    torch.from_numpy(splits["test"][0]).float(),
                    torch.from_numpy(splits["test"][1]).long(),
                ),
                "num_classes": len(le.classes_),
                "label_mapping": le,
                "layer": depth,
            }

        return layer_jobs

    def build(
        self,
        df: pd.DataFrame,
        depth: int,
        input_vectors: np.ndarray = None,
    ):
        """Build training jobs from dataset"""

        # 1. Embedding input
        # If input_vectors.npy exists, skip embedding
        if input_vectors is None:
            logger.info("Embedding input...")
            texts = df["input"].tolist()
            input_vectors : np.ndarray = self.embedding(texts)

        # 2. Integrity Check
        if len(input_vectors) != len(df):
            raise ValueError(
                f"Dimension Mismatch: DataFrame has {len(df)} rows, "
                f"but received {len(input_vectors)} vectors."
            )

        # 3. Optimized Vector Assignment
        # Using a list of numpy arrays is faster than row-by-row iteration
        df["input_vector"] = list(input_vectors)
        
        # 2. Build jobs
        all_jobs = {}
        logger.info("Building traning job...")
        with tqdm(total=depth+1, desc="Building jobs...") as pbar:
            for l in range(depth+1):
                pbar.set_description(f"Buidling job for layer {l}...")
                all_jobs.update(self._process_layer(df, l))

        logger.success(f"Orchestration complete. {len(all_jobs)} jobs ready.")
        return all_jobs
