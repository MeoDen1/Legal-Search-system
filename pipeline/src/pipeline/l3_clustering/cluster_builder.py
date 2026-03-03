import os
import json
from typing import Dict, Any

from ...utils.data_utils import get_metadata, save_metadata

from .classes import TreeNode, ClusterNode
from .clustering import construct_cluster

class ClusterBuilder:
    def __init__(
        self,
        data_path: str,
        cluster_option: str,
        cluster_depth: int,
    ):
        self.data_path = data_path
        self.silver_path = os.path.join(self.data_path, "silver")
        self.cluster_option = cluster_option
        self.cluster_depth = cluster_depth
        self.metadata = get_metadata(self.silver_path)

    def print_tree_structure(self, node: ClusterNode, indent=0):
        # Determine prefix based on depth
        prefix = "- " * indent if indent > 0 else ""
        print(f"{prefix}{node.name}")
        
        # If the node is a ClusterNode, recurse into children
        if isinstance(node, ClusterNode):
            for child in node.children:
                self.print_tree_structure(child, indent + 1)


    def serialize_hierarchy(self, node: ClusterNode) -> Dict[str, Any]:
        """Converts the Cluster/Tree structure into a JSON-serializable dict."""
        obj = {
            "uid": str(node.uid),
            "name": node.name,
            "children": []
        }
        
        if isinstance(node, ClusterNode):
            obj["children"] = [self.serialize_hierarchy(c) for c in node.children]
            
        return obj


    def __call__(self):
        metadata_path = os.path.join(self.silver_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise NameError("metadata.json file not found")

        with open(metadata_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)

        trees = []
        for uid, data in metadata["files"].items():
            trees.append(TreeNode(uid, data["name"]))

        root = construct_cluster(trees, self.cluster_depth, self.cluster_option)

        # Serialize the structure into a JSON-serializable dict
        hierarchy = self.serialize_hierarchy(root)

        # Save the hierarchy to a file
        with open(os.path.join(self.silver_path, "cluster.json"), "w", encoding="utf-8") as fp:
            json.dump(hierarchy, fp, indent=4, ensure_ascii=False)

        self.metadata["cluster"] = {
            "cluster_id": hierarchy["uid"],
            "cluster_path": os.path.join(self.silver_path, "cluster.json")
        }

        save_metadata(self.silver_path, self.metadata)
