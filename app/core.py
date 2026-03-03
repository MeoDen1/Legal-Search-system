# import uvicorn
import os
import json
import sys
import yaml
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

# Database engine must be compile in order to import
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_ENGINE_MODULE = os.path.join(
    PROJECT_DIR, "database_engine", "build"
)

sys.path.append(DATABASE_ENGINE_MODULE)

from .embedding import PreTrainedEmbedding

try:
    import database_engine
    print(database_engine.__file__)
except ImportError as e:
    raise NameError(f"Error: Could not find database_engine module in {DATABASE_ENGINE_MODULE}")


class Core:
    def __init__(
        self,
        db_cfg_path: str,
        embedding_model_name: str = "keepitreal/vietnamese-sbert",
    ):
        if not os.path.exists(db_cfg_path):
            raise FileNotFoundError("Database config path does not exist")

        with open(db_cfg_path, "r", encoding="utf-8") as fp:
            db_cfg = yaml.safe_load(fp)

        data_path = os.path.abspath(db_cfg["data_path"])
        db_path = os.path.abspath(db_cfg["db_path"])


        self.embedding_model_name = embedding_model_name
        self.database = database_engine.Database(db_path)
        self.searcher = database_engine.Searcher(self.database)
        self.embedding = PreTrainedEmbedding(embedding_model_name)

        self.thread_pool = ThreadPoolExecutor(max_workers=5)


    def _search_worker(self, query: str) -> Dict[str, Any]:
        output = self.embedding(query)

        # Convert to list of float (match with searcher.search())
        embedding = output[0].tolist() if hasattr(output[0], "tolist") else list(output[0])

        results = self.searcher.search(embedding, query)
        return results

    def search(self, query: str):
        future = self.thread_pool.submit(self._search_worker, query)
        return future.result()
