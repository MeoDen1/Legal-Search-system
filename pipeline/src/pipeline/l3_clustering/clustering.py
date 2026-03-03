import math
from typing import List

from .classes import TreeNode, ClusterNode

def _check_depth(item_cnt: int, max_children: int, depth: int = 0) -> int:
    """Internal helper to calculate tree depth."""
    if item_cnt <= 1:
        return depth
    return _check_depth(math.ceil(item_cnt / max_children), max_children, depth + 1)


def _group_balanced(items: List, max_children: int):
    """Recursively groups items into a balanced tree."""
    if len(items) <= 1:
        return items

    num_items = len(items)
    num_clusters = math.ceil(num_items / max_children)
    base_size = num_items // num_clusters
    remainder = num_items % num_clusters
    
    new_clusters = []
    cursor = 0
    for i in range(num_clusters):
        current_size = base_size + (1 if i < remainder else 0)
        chunk = items[cursor : cursor + current_size]
        new_clusters.append(ClusterNode(chunk))
        cursor += current_size
        
    return _group_balanced(new_clusters, max_children) if len(new_clusters) > 1 else new_clusters


def construct_cluster(
    trees: List[TreeNode],
    cluster_depth: int = 1,
    option: str = "max"
) -> ClusterNode:
    if cluster_depth < 1:
        raise NameError("Cluster depth must be at least 1")
    
    if len(trees) == 1:
        if cluster_depth > 1:
            raise NameError("Can not create hierarchical clusters with given depth")
        
        return ClusterNode([trees[0]])
    
    # A cluster contain at least 2 trees (excepts if only 1 tree)
    l = 2
    r = len(trees)
    val = -1

    while l <= r:
        m = (l+r) // 2
        depth = _check_depth(len(trees), m, 0)

        if depth == cluster_depth:
            # since there are multiple way to create a hierarchical_clusters
            # with `cluster_depth`
            # - option: max -> create with max clusters (each cluster will have minimum number of children)
            # - option: min -> create with min clusters (each cluster will have maximum number of children)
            val = m

            if option == "max":
                l = m + 1
            elif option == "min":
                r = m - 1

        elif depth > cluster_depth:
            l = m + 1
        else:
            r = m - 1

    if val == -1:
        raise NameError("Can not create hierarchical clusters with given depth")
    
    return _group_balanced(trees, val)[0]
