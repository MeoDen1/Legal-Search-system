import os
import shutil

class InitStorage:
    def __init__(
        self,
        data_path: str
    ):
        self.data_path = data_path

    def __call__(self, refresh: bool = False):
        """
        Initialize data storage, the `data_path` contains 3 folders: 
        `bronze`, `silver`, `gold`
        """

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if refresh:
            # delete all files in data folder
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file == "metadata.json":
                        os.remove(os.path.join(root, file))

                for dir in dirs:
                    shutil.rmtree(os.path.join(root, dir))

        for dir in ["bronze", "silver", "gold"]:
            os.makedirs(os.path.join(self.data_path, dir), exist_ok=True)
            # Create .gitkeep file (optional)
            with open(os.path.join(self.data_path, dir, ".gitkeep"), "w") as f:
                f.write("")
