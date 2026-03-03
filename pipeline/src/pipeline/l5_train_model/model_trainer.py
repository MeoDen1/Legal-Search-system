import os
import torch
import pandas as pd
from typing import Dict, Any
from loguru import logger
from .trainer import Trainer
from .training_jobs import TrainingJobs

class ModelTrainer:
    def __init__(
        self,
        dataset_path: str,
        save_path: str,
        build_depth: int,
        training_config: Dict[str, Any]
    ):
        """
        Train decoders model from the `dataset_path`
        Paramters
        -
        dataset_path: str
            Path to the dataset
        save_path: str
            Path to save the models
        build_depth: int
            Depth of the tree that decoders will be trained

        """
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.build_depth = build_depth
        self.training_jobs = TrainingJobs()
        self.trainer = Trainer(save_path, training_config)

    def __call__(self):
        df = pd.read_csv(os.path.join(self.dataset_path, "dataset.csv"))
        logger.info(f"Creating training jobs...")
        training_jobs = self.training_jobs.build(df, self.build_depth)
        logger.info(f"Training models...")
        self.trainer.train_all_jobs(training_jobs)
