"""
TemporalSampler base class — platform-agnostic.

Tạo cặp training (snapshot_t → snapshot_{t+1}) từ multi-snapshot data.
Platform-specific subclasses (twitter_pa, reddit_pa) override ENGAGEMENT_COLS
và thêm simulate_snapshots() với growth profile phù hợp.

Yêu cầu schema tối thiểu:
    post_id    : str       – ID duy nhất bài đăng
    crawl_time : datetime  – thời điểm crawl (ISO string hoặc datetime)
    + các cột trong ENGAGEMENT_COLS
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TemporalSampler:
    """
    Tạo cặp (t, t+1) cho online temporal learning.

    Subclass override ENGAGEMENT_COLS cho từng platform:
        class TwitterTemporalSampler(TemporalSampler):
            ENGAGEMENT_COLS = ["likes", "comments", "reposts", "views"]

        class RedditTemporalSampler(TemporalSampler):
            ENGAGEMENT_COLS = ["score", "num_comments", "upvote_ratio"]

    Usage
    -----
    sampler = PlatformTemporalSampler()
    df_t, df_t1 = sampler.create_pairs(df_multi)
    """

    ENGAGEMENT_COLS: list[str] = ["likes", "comments", "reposts", "views"]

    @property
    def REQUIRED_COLS(self) -> set[str]:
        return {"post_id", "crawl_time"} | set(self.ENGAGEMENT_COLS)

    def create_pairs(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Từ multi-snapshot DataFrame, tạo cặp (snapshot_t, snapshot_{t+1}).

        Mỗi hàng i trong output tương ứng với:
            df_t.iloc[i]  → input  tại bước t
            df_t1.iloc[i] → target tại bước t+1

        Chỉ giữ các post có ≥2 snapshot.

        Returns
        -------
        df_t  : DataFrame
            Cột: post_id, text, image_path, <ENGAGEMENT_COLS>,
                 delta_t_hours, crawl_time, [extra cols...]
        df_t1 : DataFrame
            Cột: <ENGAGEMENT_COLS> (renamed từ *_next)
        """
        self._validate(df)

        df = df.copy()
        df["crawl_time"] = pd.to_datetime(df["crawl_time"], errors="coerce", utc=True)
        df = df.sort_values(["post_id", "crawl_time"]).reset_index(drop=True)

        fixed_cols = self.REQUIRED_COLS | {"text", "image_path", "crawl_time"}
        extra_cols = [c for c in df.columns if c not in fixed_cols]

        rows_t: list[dict] = []
        rows_t1: list[dict] = []

        for post_id, group in df.groupby("post_id", sort=False):
            group = group.sort_values("crawl_time").reset_index(drop=True)
            if len(group) < 2:
                continue

            for i in range(len(group) - 1):
                s_t  = group.iloc[i]
                s_t1 = group.iloc[i + 1]

                delta_h = (s_t1["crawl_time"] - s_t["crawl_time"]).total_seconds() / 3600.0

                row_t: dict = {
                    "post_id":       post_id,
                    "text":          s_t.get("text", ""),
                    "image_path":    s_t.get("image_path", None),
                    "delta_t_hours": float(delta_h),
                    "crawl_time":    s_t["crawl_time"],
                }
                for col in self.ENGAGEMENT_COLS:
                    row_t[col] = float(s_t[col])
                for col in extra_cols:
                    row_t[col] = s_t.get(col, None)

                row_t1: dict = {f"{col}_next": float(s_t1[col]) for col in self.ENGAGEMENT_COLS}
                row_t1["crawl_time_next"] = s_t1["crawl_time"]
                for col in extra_cols:
                    row_t1[col] = s_t1.get(col, None)

                rows_t.append(row_t)
                rows_t1.append(row_t1)

        if not rows_t:
            raise ValueError(
                "Không tìm thấy cặp temporal nào. Mỗi post cần ≥2 snapshot. "
                "Với single-snapshot data, hãy gọi simulate_snapshots(df) trước."
            )

        df_t  = pd.DataFrame(rows_t).reset_index(drop=True)
        df_t1 = pd.DataFrame(rows_t1).reset_index(drop=True)

        # Rename *_next → tên chuẩn để TargetBuilder có thể đọc
        rename = {f"{col}_next": col for col in self.ENGAGEMENT_COLS}
        df_t1 = df_t1.rename(columns=rename)

        return df_t, df_t1

    def _validate(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"TemporalSampler: thiếu cột bắt buộc: {missing}\n"
                f"Cột hiện có: {list(df.columns)}"
            )

    def summary(self, df_t: pd.DataFrame) -> None:
        n_pairs = len(df_t)
        n_posts = df_t["post_id"].nunique()
        avg_delta = df_t["delta_t_hours"].mean()
        print(f"  Temporal pairs : {n_pairs}")
        print(f"  Unique posts   : {n_posts}  (avg {n_pairs/n_posts:.1f} pairs/post)")
        print(f"  Avg Δt         : {avg_delta:.1f}h  "
              f"[{df_t['delta_t_hours'].min():.1f}h – {df_t['delta_t_hours'].max():.1f}h]")
