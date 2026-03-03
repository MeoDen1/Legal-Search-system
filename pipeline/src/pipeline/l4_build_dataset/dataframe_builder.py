import json
import os
import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import List, Dict

class DataframeBuilder:
    def __init__(
        self,
        jsons_path: str,
        cluster_path: str
    ):
        self.jsons_path = jsons_path
        self.cluster_path = cluster_path
        self.tree_to_cluster_path = {}

    def _load_cluster_hierarchy(self):
        """Maps document UIDs to their parent cluster path from cluster.json."""
        with open(self.cluster_path, "r", encoding="utf-8") as f:
            root_data = json.load(f)

        def traverse_clusters(node, path):
            # If it's a leaf in cluster.json (a Document), store the path of parent cluster UIDs
            if "children" not in node or not node["children"]:
                self.tree_to_cluster_path[node["uid"]] = path
                return

            # Recurse through clusters, appending current cluster UID to path
            for child in node["children"]:
                traverse_clusters(child, path + [node["uid"]])

        traverse_clusters(root_data, [])

    # In order to improve the quaility of dataset's input
    # The input includes the parent's value, which provide more context, scope
    # For instance, the leaf node's value: "doanh nghiệp"
    # By providing the parent's value, the input becomes: "Luật Doanh nghiệp 2020: Đối tượng áp dụng: Doanh nghiệp"
    # Parameters: `depth`, `extract_depth` help specify which parent's value is taken for the final input
    def _extract_leaf_samples(
        self,
        node: Dict,
        parent_values: List[str],
        parent_uids: List[str],
        samples: List[Dict],
        depth: int = 0,
        extract_depth: int = 2,
    ):
        """
        Recursively finds leaf nodes.
        Labels = [Cluster IDs] + [Doc ID] + [Parent Section UIDs]
        """
        current_uid: str = node.get("uid", "")
        subitems = node.get("subitems", [])
        value = node.get("value", "").strip()

        # Update the path for children
        # We don't include the current_uid in the path yet because if THIS is the leaf,
        # the current_uid is the last element of the labels.
        path_for_children = parent_uids + [current_uid]
        value_list = parent_values
        # Adhoc: Improve the quality of the dataset's input
        if depth >= extract_depth:
            value_list = parent_values + [value.strip(":,; ")]

        if len(subitems) == 0:
            # This is a leaf node
            if value:
                samples.append(
                    {
                        "input": ": ".join(value_list),
                        "labels": path_for_children,  # This contains [Clusters... + DocID + Sections...]
                    }
                )
        else:
            # Not a leaf, keep digging
            for subitem in subitems:
                self._extract_leaf_samples(
                    subitem, value_list, path_for_children, samples, depth + 1
                )

    def __call__(self) -> pd.DataFrame:
        self._load_cluster_hierarchy()
        all_data = []

        for doc_uid, cluster_path in tqdm(self.tree_to_cluster_path.items(), desc="Building dataframe"):
            # Handle filename standardization
            safe_filename = doc_uid.replace("/", "_") + ".json"
            file_path = os.path.join(self.jsons_path, safe_filename)

            if not os.path.exists(file_path):
                logger.error(f"Warning: {file_path} not found.")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                doc_tree = json.load(f)

            leaf_samples = []
            # Start recursion: parent_uids is the cluster_path
            self._extract_leaf_samples(
                doc_tree, [doc_tree["key"]], cluster_path, leaf_samples
            )
            all_data.extend(leaf_samples)

        # Create DataFrame
        df = pd.DataFrame(all_data)

        return df
