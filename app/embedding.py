import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from loguru import logger


class PreTrainedEmbedding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SentenceTransformer automatically handles tokenizer and pooling logic
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.eval()

    def __call__(self, texts: str | List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        # 1. Sanitize Data (Prevents CUDA asserts from empty/null inputs)
        clean_texts = [str(t) if (t and str(t).strip()) else "trống" for t in texts]
        
        logger.info(
            f"APP | Embedding {len(clean_texts)} rows using SentenceTransformers on {self.device}..."
        )

        # 2. Encode
        # encode() handles batching, pooling, and moving to CPU/numpy automatically
        embeddings = self.model.encode(
            clean_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings
