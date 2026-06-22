"""
FeatureBuilder: ghép z_text và z_image thành vector đầu vào z = [z_text ; z_image].

Ngoài embedding, builder còn có thể thêm metadata features (optional)
như hour_of_day, day_of_week, has_image flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ...utils.text_encoder import TextEncoder
from .image_encoder import ImageEncoder


class FeatureBuilder:
    """
    Xây dựng feature matrix z cho từng post.

    Pipeline:
      1. Encode text  → z_text  ∈ R^{text_dim}
      2. Encode image → z_image ∈ R^{image_dim}   (zeros nếu không có ảnh)
      3. (Optional) metadata features → z_meta ∈ R^{3}  (has_image, hour, dow)
      4. (Optional) engagement features → z_eng ∈ R^{5}
             [log1p(likes_t), log1p(comments_t), log1p(reposts_t),
              log1p(views_t), log1p(delta_t_hours+1)]
         Cần thiết cho temporal online learning:
         model nhìn thấy trạng thái hiện tại để dự đoán trạng thái tiếp theo.
      5. Concat → z = [z_text ; z_image ; z_meta ; z_eng]
      6. Chuẩn hóa bằng StandardScaler

    Parameters
    ----------
    text_encoder : TextEncoder
    image_encoder : ImageEncoder
    text_col : str
    image_col : str
    add_metadata : bool
        Thêm has_image, hour_of_day, day_of_week.
    add_engagement : bool
        Thêm engagement tại thời điểm t làm feature (dùng cho temporal learning).
        Khi True, df phải có cột: likes, comments, reposts, views.
        Và tùy chọn: delta_t_hours (thời gian kể từ snapshot trước).
    delta_t_col : str
        Tên cột chứa khoảng cách thời gian giữa 2 snapshot (giờ).
    scale : bool
    """

    METADATA_COLS = ["hour_of_day", "day_of_week", "has_image"]
    ENGAGEMENT_COLS = ["likes", "comments", "reposts", "views"]
    # Author metadata — tín hiệu virality rất mạnh (median follower Low=186 -> Viral=310k).
    # Các cột này là HẰNG SỐ theo post (lặp lại trên mọi snapshot của post).
    AUTHOR_COLS = [
        "author_log_followers", "author_blue_verified", "author_verified",
        "author_followers_per_day", "author_ff_ratio", "author_age_days",
    ]

    def __init__(
        self,
        text_encoder: TextEncoder | None = None,
        image_encoder: ImageEncoder | None = None,
        text_col: str = "text",
        image_col: str = "image_path",
        add_metadata: bool = True,
        add_engagement: bool = False,
        add_author: bool = False,
        add_age: bool = False,
        delta_t_col: str = "delta_t_hours",
        age_col: str = "age_h",
        side_scale: float = 1.0,
        scale: bool = True,
    ):
        self.text_encoder = text_encoder or TextEncoder(backend="tfidf", text_dim=768)
        self.image_encoder = image_encoder or ImageEncoder(backend="zeros", image_dim=512)
        self.text_col = text_col
        self.image_col = image_col
        self.add_metadata = add_metadata
        self.add_engagement = add_engagement
        self.add_author = add_author
        self.add_age = add_age
        self.delta_t_col = delta_t_col
        self.age_col = age_col
        self.scale = scale
        # side_scale: nhân các feature "phụ" (meta/engagement/age/author) SAU khi z-score
        # để chúng không bị ~2176 chiều embedding nhấn chìm trong ||x||^2 của PA.
        # (vẫn full-linear; chỉ là hệ số tỉ lệ cố định)
        self.side_scale = side_scale

        self.scaler = StandardScaler()
        self._fitted = False
        self.feature_dim: int = 0
        self._n_embed: int = 0          # bề rộng khối [text; image] (để áp side_scale)
        # Optional precomputed embeddings mapping: post_id -> np.ndarray
        self._text_embeddings: dict | None = None
        self._image_embeddings: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit encoders + scaler và transform toàn bộ DataFrame.
        Gọi lần đầu trên training data.

        Returns: z matrix shape (n, feature_dim)
        """
        z = self._build_raw(df, fit=True)
        if self.scale:
            z = self.scaler.fit_transform(z)
        z = self._apply_side_scale(z)
        self._fitted = True
        self.feature_dim = z.shape[1]
        return z.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform mà không fit lại. Dùng cho inference / online update.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        z = self._build_raw(df, fit=False)
        if self.scale:
            z = self.scaler.transform(z)
        z = self._apply_side_scale(z)
        return z.astype(np.float32)

    def transform_single(self, row: dict | pd.Series) -> np.ndarray:
        """
        Transform một post đơn lẻ → shape (1, feature_dim).
        Dùng trong online inference.
        """
        df = pd.DataFrame([row])
        return self.transform(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_raw(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        """Builds unnormalized feature matrix."""
        parts: list[np.ndarray] = []

        # --- Text features ---
        texts = df[self.text_col].fillna("").tolist() if self.text_col in df.columns else [""] * len(df)
        # If precomputed text embeddings provided, use them keyed by post_id
        z_text = None
        if self._text_embeddings is not None:
            ids = df["post_id"].astype(str).tolist() if "post_id" in df.columns else [str(i) for i in df.index]
            emb_list = []
            missing_texts = []
            for i, pid in enumerate(ids):
                emb = self._text_embeddings.get(pid)
                if emb is None:
                    emb_list.append(None)
                    missing_texts.append((i, texts[i]))
                else:
                    emb_list.append(np.asarray(emb, dtype=np.float32))

            # Fit encoder on missing texts if needed
            if fit and missing_texts:
                unique_missing = list(dict.fromkeys([t for _, t in missing_texts]))
                self.text_encoder.fit(unique_missing)

            # Build final z_text, filling missing entries via text_encoder
            encoded_missing = None
            if missing_texts:
                missing_texts_only = [t for _, t in missing_texts]
                encoded_missing = self.text_encoder.encode(missing_texts_only)

            miss_idx = 0
            z_rows = []
            for item in emb_list:
                if item is None:
                    z_rows.append(encoded_missing[miss_idx])
                    miss_idx += 1
                else:
                    z_rows.append(item)
            z_text = np.stack(z_rows)
        else:
            if fit:
                # Deduplicate để TF-IDF IDF không bị lệch khi temporal data có nhiều bản sao
                unique_texts = list(dict.fromkeys(texts))
                self.text_encoder.fit(unique_texts)
            z_text = self.text_encoder.encode(texts)
        parts.append(z_text)

        # --- Image features ---
        # --- Image features ---
        # Support precomputed image embeddings mapping: post_id -> embedding or list of embeddings
        if self._image_embeddings is not None:
            ids = df["post_id"].astype(str).tolist() if "post_id" in df.columns else [str(i) for i in df.index]
            z_rows = []
            for i, pid in enumerate(ids):
                emb = self._image_embeddings.get(pid)
                if emb is None:
                    # fallback: try to use image paths with encoder
                    p = df[self.image_col].iloc[i] if self.image_col in df.columns else None
                    z_rows.append(self.image_encoder.encode([p])[0])
                else:
                    arr = np.asarray(emb, dtype=np.float32)
                    if arr.ndim == 2:
                        # multiple image embeddings -> mean pooling
                        arr = arr.mean(axis=0)
                    z_rows.append(arr)
            z_image = np.stack(z_rows)
        else:
            if self.image_col in df.columns:
                img_paths = df[self.image_col].tolist()
            else:
                img_paths = [None] * len(df)
            if fit:
                self.image_encoder.fit(img_paths)
            z_image = self.image_encoder.encode(img_paths)
        parts.append(z_image)

        # Bề rộng khối embedding [text; image] — phần còn lại là "side features"
        self._n_embed = int(sum(p.shape[1] for p in parts))

        # --- Metadata features (optional) ---
        if self.add_metadata:
            meta = self._extract_metadata(df)
            parts.append(meta)

        # --- Engagement features (temporal online learning) ---
        if self.add_engagement:
            z_eng = self._extract_engagement(df)
            parts.append(z_eng)

        # --- Absolute snapshot age (timestep) ---
        if self.add_age:
            parts.append(self._extract_age(df))

        # --- Author metadata (strong virality signal) ---
        if self.add_author:
            parts.append(self._extract_author(df))

        return np.hstack(parts)

    def _apply_side_scale(self, z: np.ndarray) -> np.ndarray:
        """Nhân khối feature phụ (sau embedding) với side_scale để tăng trọng số."""
        if self.side_scale != 1.0 and 0 < self._n_embed < z.shape[1]:
            z = z.copy()
            z[:, self._n_embed:] *= self.side_scale
        return z

    def _extract_age(self, df: pd.DataFrame) -> np.ndarray:
        """log1p tuổi tuyệt đối của snapshot (giờ) — cho model biết đang ở mốc nào."""
        n = len(df)
        a = np.zeros((n, 1), dtype=np.float32)
        if self.age_col in df.columns:
            a[:, 0] = np.log1p(
                pd.to_numeric(df[self.age_col], errors="coerce").fillna(0).clip(lower=0).values
            )
        return a

    def _extract_author(self, df: pd.DataFrame) -> np.ndarray:
        """Khối metadata tác giả (hằng số theo post). Cột thiếu -> 0."""
        n = len(df)
        A = np.zeros((n, len(self.AUTHOR_COLS)), dtype=np.float32)
        for i, c in enumerate(self.AUTHOR_COLS):
            if c in df.columns:
                A[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(0).values
        return A

    def _extract_metadata(self, df: pd.DataFrame) -> np.ndarray:
        """Returns small metadata matrix: has_image, hour_of_day, day_of_week."""
        n = len(df)
        meta = np.zeros((n, 3), dtype=np.float32)

        # has_image
        if self.image_col in df.columns:
            has_img = df[self.image_col].notna().astype(float).values
        else:
            has_img = np.zeros(n)
        meta[:, 0] = has_img

        # hour_of_day
        if "created_at" in df.columns:
            dt = pd.to_datetime(df["created_at"], errors="coerce")
            meta[:, 1] = dt.dt.hour.fillna(0).values
        elif "hour_of_day" in df.columns:
            meta[:, 1] = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(0).values

        # day_of_week
        if "created_at" in df.columns:
            dt = pd.to_datetime(df["created_at"], errors="coerce")
            meta[:, 2] = dt.dt.dayofweek.fillna(0).values
        elif "day_of_week" in df.columns:
            meta[:, 2] = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0).values

        return meta

    def _extract_engagement(self, df: pd.DataFrame) -> np.ndarray:
        """
        6 features đại diện cho trạng thái engagement hiện tại:
            [log1p(likes_t), log1p(comments_t), log1p(reposts_t), log1p(views_t),
             popularity_score_t,                 # ln(0.01V + L + 5C + 10R + 1)
             log1p(delta_t_hours + 1)]

        popularity_score_t là tín hiệu autoregressive MẠNH NHẤT cho nhãn kế tiếp
        (nhãn = phân vị của score; score ít nhảy bậc giữa 2 snapshot liền kề).
        Tính từ engagement ĐÃ quan sát tại t -> hợp lệ, không leak target.
        """
        n = len(df)
        z = np.zeros((n, 6), dtype=np.float32)

        eng = {}
        for i, col in enumerate(self.ENGAGEMENT_COLS):
            vals = (pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).values
                    if col in df.columns else np.zeros(n))
            eng[col] = vals
            z[:, i] = np.log1p(vals)

        # current popularity score (cùng công thức TargetBuilder / Label)
        z[:, 4] = np.log(
            0.01 * eng["views"] + eng["likes"]
            + 5.0 * eng["comments"] + 10.0 * eng["reposts"] + 1.0
        )

        # delta_t: log1p(hours + 1)
        if self.delta_t_col in df.columns:
            dt = pd.to_numeric(df[self.delta_t_col], errors="coerce").fillna(0).clip(lower=0).values
            z[:, 5] = np.log1p(dt + 1)

        return z

    # ------------------------------------------------------------------
    # Dimension info
    # ------------------------------------------------------------------

    @property
    def text_dim(self) -> int:
        return self.text_encoder.text_dim

    @property
    def image_dim(self) -> int:
        return self.image_encoder.image_dim

    @property
    def engagement_dim(self) -> int:
        return 6 if self.add_engagement else 0

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def set_text_embeddings(self, mapping: dict) -> None:
        """Provide a dict post_id -> embedding (np.ndarray)."""
        self._text_embeddings = {str(k): np.asarray(v, dtype=np.float32) for k, v in mapping.items()}

    def set_image_embeddings(self, mapping: dict) -> None:
        """Provide a dict post_id -> embedding or list-of-embeddings."""
        self._image_embeddings = {}
        for k, v in mapping.items():
            arr = np.asarray(v, dtype=np.float32)
            self._image_embeddings[str(k)] = arr

    def load_embeddings_npz(self, path: str, kind: str = "text") -> None:
        """Load embeddings from a .npz file containing 'ids' and 'embeds'.

        Example: np.savez('emb.npz', ids=ids_array, embeds=emb_array)
        ids : array of strings
        embeds: array shape (n, dim) or (n, k, dim) for multiple images
        """
        data = np.load(path, allow_pickle=True)
        ids = [str(x) for x in data["ids"]]
        embeds = data["embeds"]
        mapping = {ids[i]: embeds[i] for i in range(len(ids))}
        if kind == "text":
            self.set_text_embeddings(mapping)
        else:
            self.set_image_embeddings(mapping)
