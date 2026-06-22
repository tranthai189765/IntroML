"""
Image Encoder for Twitter/X posts.

Backends:
  'zeros'  – vector 0 cho mọi ảnh, dùng khi không có ảnh / test nhanh.
  'pixel'  – resize ảnh về 16×16, flatten, PCA → image_dim. Nhẹ, không GPU.
  'resnet' – ResNet50 pretrained (cần torchvision + torch).
  'clip'   – OpenAI CLIP ViT (cần pip install clip hoặc open-clip-torch).
  'precomputed' – nhận .npy file đã tính trước.

Nếu bài đăng không có ảnh (image_url = None / NaN), encoder tự trả về
vector 0 thay cho embedding ảnh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


_ZERO_SENTINEL = "__NO_IMAGE__"


class ImageEncoder:
    """
    Chuyển ảnh (đường dẫn file hoặc URL) thành vector z_image ∈ R^{image_dim}.

    Parameters
    ----------
    backend : str
        'zeros' | 'pixel' | 'resnet' | 'clip' | 'precomputed'
    image_dim : int
        Chiều output mong muốn.
    cache_dir : str | None
        Thư mục lưu embedding đã tính (data/embeddings/).
    """

    def __init__(
        self,
        backend: str = "zeros",
        image_dim: int = 512,
        cache_dir: str | None = None,
    ):
        self.backend = backend
        self.image_dim = image_dim
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._pca = None
        self._resnet = None
        self._clip_model = None
        self._clip_preprocess = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, image_paths: list[str | None]) -> "ImageEncoder":
        """Fit PCA for 'pixel' backend; load models for 'resnet'/'clip'."""
        if self.backend == "pixel":
            self._fit_pixel(image_paths)
        elif self.backend == "resnet":
            self._load_resnet()
        elif self.backend == "clip":
            self._load_clip()
        self._fitted = True
        return self

    def encode(self, image_paths: list[str | None] | pd.Series) -> np.ndarray:
        """
        Returns embedding matrix of shape (n, image_dim).
        None / NaN entries → zero vector.
        """
        if isinstance(image_paths, pd.Series):
            image_paths = image_paths.tolist()

        if not self._fitted:
            self.fit(image_paths)

        if self.backend == "zeros":
            return np.zeros((len(image_paths), self.image_dim), dtype=np.float32)
        elif self.backend == "pixel":
            return self._encode_pixel(image_paths)
        elif self.backend == "resnet":
            return self._encode_resnet(image_paths)
        elif self.backend == "clip":
            return self._encode_clip(image_paths)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def encode_from_file(self, path: str) -> np.ndarray:
        """Loads pre-computed image embeddings from a .npy file."""
        return np.load(path).astype(np.float32)

    # ------------------------------------------------------------------
    # Pixel backend (nhẹ, không cần GPU)
    # ------------------------------------------------------------------

    def _load_image_rgb(self, path: str | None) -> np.ndarray | None:
        """Returns (H, W, 3) uint8 or None if path is invalid."""
        if path is None or (isinstance(path, float) and np.isnan(path)):
            return None
        try:
            from PIL import Image
            img = Image.open(str(path)).convert("RGB").resize((16, 16))
            return np.array(img, dtype=np.float32) / 255.0
        except Exception:
            return None

    def _fit_pixel(self, image_paths: list[str | None]) -> None:
        from sklearn.decomposition import PCA

        raw = []
        for p in image_paths:
            img = self._load_image_rgb(p)
            raw.append(img.flatten() if img is not None else np.zeros(16 * 16 * 3, dtype=np.float32))

        raw_arr = np.stack(raw)
        n_components = min(self.image_dim, raw_arr.shape[1], len(raw_arr) - 1)
        self._pca = PCA(n_components=n_components, random_state=42)
        self._pca.fit(raw_arr)
        self.image_dim = n_components

    def _encode_pixel(self, image_paths: list[str | None]) -> np.ndarray:
        raw = []
        for p in image_paths:
            img = self._load_image_rgb(p)
            raw.append(img.flatten() if img is not None else np.zeros(16 * 16 * 3, dtype=np.float32))
        raw_arr = np.stack(raw)
        return self._pca.transform(raw_arr).astype(np.float32)

    # ------------------------------------------------------------------
    # ResNet backend
    # ------------------------------------------------------------------

    def _load_resnet(self) -> None:
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove final FC layer → output 2048-dim pool features
            self._resnet = torch.nn.Sequential(*list(model.children())[:-1])
            self._resnet.eval().to(self._device)
            self.image_dim = 2048

            self._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        except ImportError:
            raise ImportError("torch + torchvision required. pip install torch torchvision")

    def _encode_resnet(self, image_paths: list[str | None]) -> np.ndarray:
        import torch
        from PIL import Image

        embeddings = []
        for p in image_paths:
            if p is None or (isinstance(p, float) and np.isnan(p)):
                embeddings.append(np.zeros(self.image_dim, dtype=np.float32))
                continue
            try:
                img = Image.open(str(p)).convert("RGB")
                tensor = self._transform(img).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = self._resnet(tensor).squeeze().cpu().numpy()
                embeddings.append(feat.astype(np.float32))
            except Exception:
                embeddings.append(np.zeros(self.image_dim, dtype=np.float32))
        return np.stack(embeddings)

    # ------------------------------------------------------------------
    # CLIP backend
    # ------------------------------------------------------------------

    def _load_clip(self) -> None:
        try:
            import open_clip
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self._clip_model.eval()
            self.image_dim = 512
        except ImportError:
            raise ImportError("open-clip-torch required. pip install open-clip-torch")

    def _encode_clip(self, image_paths: list[str | None]) -> np.ndarray:
        import torch
        from PIL import Image

        embeddings = []
        for p in image_paths:
            if p is None or (isinstance(p, float) and np.isnan(p)):
                embeddings.append(np.zeros(self.image_dim, dtype=np.float32))
                continue
            try:
                img = self._clip_preprocess(Image.open(str(p)).convert("RGB")).unsqueeze(0)
                with torch.no_grad():
                    feat = self._clip_model.encode_image(img).squeeze().numpy()
                embeddings.append(feat.astype(np.float32))
            except Exception:
                embeddings.append(np.zeros(self.image_dim, dtype=np.float32))
        return np.stack(embeddings)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_embeddings(self, embeddings: np.ndarray, name: str) -> Path:
        if self.cache_dir is None:
            raise ValueError("Set cache_dir to save embeddings.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{name}.npy"
        np.save(path, embeddings)
        return path

    def load_embeddings(self, name: str) -> np.ndarray:
        path = self.cache_dir / f"{name}.npy"
        return np.load(path).astype(np.float32)
