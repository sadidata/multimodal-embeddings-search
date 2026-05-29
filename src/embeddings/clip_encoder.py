"""CLIP-based multimodal encoder for image and text embeddings."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Union

import clip
import numpy as np
import torch
from PIL import Image


class CLIPEncoder:
    """Encodes images and text into a shared CLIP embedding space.

    Supports batch encoding with GPU acceleration and result caching.

    Example::

        encoder = CLIPEncoder(model="ViT-B/32", device="cuda")
        img_emb = encoder.encode_image("photo.jpg")          # (512,)
        txt_emb = encoder.encode_text("a red sports car")     # (512,)
        sim = encoder.cosine_similarity(img_emb, txt_emb)
    """

    AVAILABLE_MODELS = [
        "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", "RN50x4",
    ]

    def __init__(
                         self,
                         model: str = "ViT-B/32",
                         device: Optional[str] = None,
        cache_size: int = 2048,
    ):
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model}. Choose from {self.AVAILABLE_MODELS}")

        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._preprocess = clip.load(model, device=self.device)
        self._model.eval()
        self.embedding_dim = self._model.visual.output_dim
        self._cache_size = cache_size

    @property
    def dimension(self) -> int:
        """Embedding dimension for this model."""
        return self.embedding_dim

    # ------------------------------------------------------------------
    # Encoding methods
    # ------------------------------------------------------------------

    def encode_image(
                             self,
                             source: Union[str, Path, Image.Image],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a single image.

        Args:
            source: File path or PIL Image.
            normalize: L2-normalize the embedding (recommended for cosine search).

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        if isinstance(source, (str, Path)):
            image = Image.open(source).convert("RGB")
        else:
            image = source.convert("RGB")

        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        emb = features.cpu().numpy().squeeze()
        return emb / np.linalg.norm(emb) if normalize else emb

    def encode_images_batch(
                                    self,
                                    sources: List[Union[str, Path, Image.Image]],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a batch of images efficiently.

        Args:
            sources: List of file paths or PIL Images.
            batch_size: GPU batch size.
            normalize: L2-normalize embeddings.

        Returns:
            2D numpy array of shape (n, embedding_dim).
        """
        all_embeddings = []

        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]
            tensors = []
            for src in batch:
                if isinstance(src, (str, Path)):
                    img = Image.open(src).convert("RGB")
                else:
                    img = src.convert("RGB")
                tensors.append(self._preprocess(img))

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                features = self._model.encode_image(batch_tensor)
            embeddings = features.cpu().numpy()
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def encode_text(
                            self,
                            text: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text query or list of texts.

        Args:
            text: Query string or list of strings.
            normalize: L2-normalize embeddings.

        Returns:
            1D or 2D numpy array of embeddings.
        """
        single = isinstance(text, str)
        texts = [text] if single else text

        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        embeddings = features.cpu().numpy()

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings[0] if single else embeddings

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized embeddings."""
        return float(np.dot(a, b))

    @staticmethod
    def batch_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query and a corpus matrix."""
        return corpus @ query

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
                            f"CLIPEncoder(model={self.model_name!r}, "
                            f"device={self.device!r}, "
                            f"dim={self.embedding_dim})"
                        )
