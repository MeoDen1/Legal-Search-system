import os
import sys

DATABASE_ENGINE_MODULE = os.getenv("DATABASE_ENGINE_MODULE")

if not DATABASE_ENGINE_MODULE:
    raise ValueError("DATABASE_ENGINE_MODULE environment variable is not set")

sys.path.append(DATABASE_ENGINE_MODULE)

try:
    import database_engine
    print(database_engine.__file__)
except ImportError as e:
    raise NameError(f"Error: Could not find database_engine module in {DATABASE_ENGINE_MODULE}")

class DatabaseBuilder:
    def __init__(
        self,
        data_path: str,
        database_path: str
    ):
        self.data_path = data_path
        self.gold_data = os.path.join(self.data_path, "gold")
        self.database_path = database_path

        self.ingestion_manager = database_engine.IngestionManager(self.gold_data, self.database_path)

    def __call__(self):
        self.ingestion_manager.ingest()