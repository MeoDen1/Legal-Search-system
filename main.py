import os
import yaml
from loguru import logger
from dotenv import load_dotenv
from pipeline import Pipeline
from app import Core

CONFIG_PATH = "configs"
ENV_PATH = ".env"

load_dotenv(ENV_PATH, override=True)

if __name__ == "__main__":
    logger.level("INFO")
    pipeline_cfg_path = os.path.join(CONFIG_PATH, "pipeline_config.yaml")
    training_cfg_path = os.path.join(CONFIG_PATH, "training_config.yaml")
    database_cfg_path = os.path.join(CONFIG_PATH, "database_config.yaml")
    pipeline = Pipeline(pipeline_cfg_path, training_cfg_path, database_cfg_path, ENV_PATH)
    pipeline()