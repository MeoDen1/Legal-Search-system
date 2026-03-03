import os
import shutil
import time
import json
import yaml
import datetime
import subprocess
from dotenv import load_dotenv
from loguru import logger

from pathlib import Path
from .src.pipeline.l0_init_storage import InitStorage
from .src.pipeline.l1_ingestion import Ingestion
from .src.pipeline.l2_processing import Processor
from .src.pipeline.l3_clustering import ClusterBuilder
from .src.pipeline.l4_build_dataset import DatasetBuilder
from .src.pipeline.l5_train_model import ModelTrainer
from .src.pipeline.l6_init_database import DatabaseBuilder

PIPELINE_SRC = os.path.dirname(os.path.abspath(__file__))


class Pipeline:
    def __init__(
        self,
        pipeline_config_path: str,
        training_config_path: str,
        database_config_path: str,
        env_path: str
    ):
        load_dotenv(env_path, override=True)

        with open(pipeline_config_path, "r") as fp:
            config = yaml.safe_load(fp)

        with open(training_config_path, "r") as fp:
            config["training_config"] = yaml.safe_load(fp)
        
        with open(database_config_path, "r") as fp:
            config["database_config"] = yaml.safe_load(fp)

        self.config = config
        self.env_path = env_path
        self.l5_train_model_script = os.path.join(PIPELINE_SRC, "scripts", "l5_train_model.sh")
        self.l5_train_model_notebook = os.path.join(PIPELINE_SRC, "notebooks")

        self.data_path = os.path.abspath(config["data_path"])
        self.src_path = os.path.abspath(config["src_path"])

        self.bronze_path = os.path.join(self.data_path, "bronze")
        self.silver_path = os.path.join(self.data_path, "silver")
        self.gold_path = os.path.join(self.data_path, "gold")

        self.dataset_dir = os.path.join(self.silver_path, "datasets")

        self.training_config_path = training_config_path

        for key in self.config:
            if not isinstance(self.config[key], dict):
                continue

            self.config[key]["data_path"] = self.data_path


    def __call__(self):
        fresh_init = self.config.get("fresh_init", False)
        step = self.config.get("start", "l0")
        training_kernel = self.config.get("training_kernel", "local")

        KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
        if training_kernel == "kaggle" and KAGGLE_USERNAME is None:
            raise ValueError("KAGGLE_USERNAME is not set")

        if "l0" >= step:
            logger.info(f"Initializing data storage at {self.data_path}")
            init_storage = InitStorage(self.data_path)
            init_storage(refresh=fresh_init)

        # if "l2" >= step:
        #     logger.info("Processing data...")
        #     document_tree_depth = int(self.config["processing"]["document_tree_depth"])
        #     processor = Processor(self.data_path, document_tree_depth)
        #     processor()
        #     logger.success("Processing completed. Data is saved at {}".format(self.silver_path))
        #

        if "l3" >= step:
            logger.info("Clustering document...")
            cluster_option = self.config["clustering"]["cluster_option"]
            cluster_depth = self.config["clustering"]["cluster_depth"]
            cluster_builder = ClusterBuilder(self.data_path, cluster_option, cluster_depth)
            cluster_builder()
            logger.success("Clustering done. clustering.json is saved at {}".format(self.silver_path))


        if "l4" >= step:
            logger.info("Building dataset...")
            dataset_builder = DatasetBuilder(self.data_path)
            df = dataset_builder()
            logger.success("Dataset is created with {} samples".format(len(df)))

        if "l5" >= step:
            logger.info("Training model...")

            if training_kernel == "local":
                training_config = self.config["training_config"]
                build_depth = training_config.get("build_depth", 1)
                dataset_path = os.path.join(self.silver_path, "datasets")
                model_trainer = ModelTrainer(dataset_path, self.gold_path, build_depth, training_config)
                model_trainer()
                logger.success("Model training done")

            elif training_kernel == "kaggle":
                from .src.utils.kaggle_utils import SRC_IGNORE_PATTERNS, upload_to_kaggle
                # TODO
                # - Better check method to ensure dataset & src_code are uploaded
                # - Change to git for stable release

                # Copy training_cfg to datasets
                logger.info("Uploading dataset to kaggle...")
                shutil.copyfile(
                    self.training_config_path,
                    os.path.join(self.silver_path, "datasets", "training_config.yaml")
                )
                upload_to_kaggle(
                    path=os.path.join(self.silver_path, "datasets"),
                    name="legal-sys-decoder-dataset",
                    username=KAGGLE_USERNAME
                )

                logger.info("Uploading src code to kaggle...")

                upload_to_kaggle(
                    path=self.src_path,
                    name="legal-sys-src",
                    username=KAGGLE_USERNAME,
                    ignore_patterns=SRC_IGNORE_PATTERNS
                )
        
                # Ensure dataset & src code are appeared in kaggle
                # (even though they are already uploaded, it takes time for kaggle to acknowledge the dataset)
                time.sleep(10)
                # Run training script
                with subprocess.Popen(
                    [
                        "bash", str(self.l5_train_model_script),
                        f"{self.l5_train_model_notebook}", 
                        f"{self.gold_path}",
                        f"{self.env_path}"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, # Redirect stderr to stdout to catch all logs
                    text=True,
                    bufsize=1,                # Line-buffered for real-time output
                    universal_newlines=True
                ) as process:
                    
                    # Read output line by line as it is generated
                    for line in process.stdout:
                        logger.bind(task="l5_train_model_script").info(f"{line.strip()}")

            # Extract all document_jsons + cluster.json from silver to gold
            shutil.copytree(
                os.path.join(self.silver_path, "document_jsons"),
                os.path.join(self.gold_path, "document_jsons"),
                dirs_exist_ok=True
            )

            shutil.copy(
                os.path.join(self.silver_path, "cluster.json"),
                os.path.join(self.gold_path, "cluster.json"),
            )

            logger.success("Model training done")

        if "l6" >= step:
            logger.info("Initialize Database engine...")
            database_path = self.config["database_config"]["db_path"]
            database_builder = DatabaseBuilder(self.data_path, database_path)
            database_builder()
            logger.success("Database engine initialized")
            
