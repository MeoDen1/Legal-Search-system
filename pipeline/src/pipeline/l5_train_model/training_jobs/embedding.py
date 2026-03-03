import torch
import numpy as np
from tqdm import tqdm
from typing import List
from loguru import logger
from sentence_transformers import SentenceTransformer

class PretrainedEmbedding:
    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SentenceTransformer automatically handles tokenizer and pooling logic
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.eval()

    def __call__(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        # 1. Sanitize Data (Prevents CUDA asserts from empty/null inputs)
        clean_texts = [str(t) if (t and str(t).strip()) else "trống" for t in texts]
        
        logger.info(
            f"Embedding {len(clean_texts)} rows using SentenceTransformers on {self.device}..."
        )

        # 2. Encode
        # encode() handles batching, pooling, and moving to CPU/numpy automatically
        embeddings = self.model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True # Recommended for SBERT cosine similarity
        )

        return embeddings