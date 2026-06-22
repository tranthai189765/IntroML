"""
Text Encoder for Twitter/X posts.

Backends (chọn qua backend= khi khởi tạo):
  'tfidf'  – TF-IDF + Truncated SVD, không cần GPU, dùng để test/demo.
  'sbert'  – sentence-transformers (cần pip install sentence-transformers).
  'precomputed' – nhận embedding numpy đã tính trước, bỏ qua encode().

Output luôn là numpy array shape (n, text_dim).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


class TextEncoder:
    """
    Chuyển cột text thành vector z_text ∈ R^{text_dim}.

    Parameters
    ----------
    backend : str
        'tfidf' | 'sbert' | 'precomputed'
    text_dim : int
        Chiều output mong muốn.
        - 'tfidf': chiều SVD (mặc định 768).
        - 'sbert': phụ thuộc model (all-MiniLM-L6-v2 → 384,
                   all-mpnet-base-v2 → 768).
        - 'precomputed': không dùng.
    sbert_model : str
        Tên model sentence-transformers nếu backend='sbert'.
    cache_dir : str | None
        Thư mục lưu embedding đã tính (data/embeddings/).
    """

    def __init__(
        self,
        backend: str = "tfidf",
        text_dim: int = 768,
        sbert_model: str = "all-MiniLM-L6-v2",
        cache_dir: str | None = None,
    ):
        self.backend = backend
        self.text_dim = text_dim
        self.sbert_model_name = sbert_model
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._tfidf = None
        self._svd = None
        self._sbert = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> "TextEncoder":
        """Fits the encoder on a list of strings (only needed for 'tfidf')."""
        if self.backend == "tfidf":
            self._fit_tfidf(texts)
        elif self.backend == "sbert":
            self._load_sbert()
        self._fitted = True
        return self

    def encode(self, texts: list[str] | pd.Series) -> np.ndarray:
        """
        Returns embedding matrix of shape (len(texts), text_dim).
        Calls fit() first if not yet fitted.
        """
        if isinstance(texts, pd.Series):
            texts = texts.fillna("").tolist()
        texts = [str(t) for t in texts]

        if not self._fitted:
            self.fit(texts)

        if self.backend == "tfidf":
            return self._encode_tfidf(texts)
        elif self.backend == "sbert":
            return self._encode_sbert(texts)
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use fit() then encode(), or use 'precomputed'.")

    def encode_from_file(self, path: str) -> np.ndarray:
        """Loads pre-computed embeddings from a .npy file."""
        arr = np.load(path)
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # TF-IDF backend
    # ------------------------------------------------------------------

    def _fit_tfidf(self, texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._tfidf = TfidfVectorizer(
            max_features=20_000,
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
        )
        tfidf_matrix = self._tfidf.fit_transform(texts)

        n_components = min(self.text_dim, tfidf_matrix.shape[1] - 1, len(texts) - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf_matrix)

        self.text_dim = n_components

    def _encode_tfidf(self, texts: list[str]) -> np.ndarray:
        tfidf_matrix = self._tfidf.transform(texts)
        emb = self._svd.transform(tfidf_matrix).astype(np.float32)
        return emb

    # ------------------------------------------------------------------
    # Sentence-BERT backend
    # ------------------------------------------------------------------

    def _load_sbert(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self.sbert_model_name)
            # Infer actual output dim from model
            test_emb = self._sbert.encode(["test"], show_progress_bar=False)
            self.text_dim = test_emb.shape[1]
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers\n"
                "Or use backend='tfidf' for lightweight encoding."
            )

    def _encode_sbert(self, texts: list[str]) -> np.ndarray:
        return self._sbert.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 500,
            convert_to_numpy=True,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_embeddings(self, embeddings: np.ndarray, name: str) -> Path:
        """Saves embedding array to cache_dir/{name}.npy."""
        if self.cache_dir is None:
            raise ValueError("Set cache_dir to save embeddings.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{name}.npy"
        np.save(path, embeddings)
        return path

    def load_embeddings(self, name: str) -> np.ndarray:
        """Loads embedding array from cache_dir/{name}.npy."""
        path = self.cache_dir / f"{name}.npy"
        return np.load(path).astype(np.float32)
