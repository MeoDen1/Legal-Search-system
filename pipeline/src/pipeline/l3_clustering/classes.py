import uuid

class TreeNode:
    def __init__(
        self,
        uid: str,
        doc_name: str=""
    ):
        self.uid = uid
        self.name = doc_name
        self.vector = None

class ClusterNode:
    def __init__(self, children=None):
        self.uid = str(uuid.uuid4())
        self.name = f"cluster-{self.uid}"
        self.children = children or []
        self.vector = None
