"""
Reddit-specific TemporalSampler.

ENGAGEMENT_COLS = ["score", "num_comments", "upvote_ratio"]

Growth profile Reddit: peak nhanh hơn Twitter (front page effect).
  - Phần lớn upvotes trong 2-6h đầu
  - Comments tiếp tục vào đến 24h
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...utils.temporal_sampler import TemporalSampler as _BaseTemporalSampler


class RedditTemporalSampler(_BaseTemporalSampler):
    """
    Reddit TemporalSampler.

    ENGAGEMENT_COLS = ["score", "num_comments", "upvote_ratio"]

    Thêm simulate_snapshots() với Reddit viral growth curve.
    Reddit front page spike: 5–15% tại t=0 → 100% tại t=24h,
    với peak growth rate mạnh hơn Twitter trong 2-4h đầu.
    """

    ENGAGEMENT_COLS = ["score", "num_comments", "upvote_ratio"]

    # upvote_ratio là tỉ lệ nên xử lý riêng (không scale theo ratio mà thêm noise nhỏ)
    _RATIO_COL = "upvote_ratio"

    @staticmethod
    def simulate_snapshots(
        df: pd.DataFrame,
        crawl_time_col: str = "created_utc",
        growth_profiles: list[dict] | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Từ DataFrame 1-snapshot/post, sinh ra multi-snapshot giả lập.

        Growth model Reddit (11 checkpoints):
            t=0h   : 5%–15%   (vừa đăng, upvotes ban đầu)
            t=0.5h : 15%–35%  (front page spike)
            t=1h   : 30%–55%  (peak viral window)
            t=1.5h : 45%–65%
            t=2h   : 55%–72%  (chậm dần)
            t=3h   : 65%–80%
            t=4h   : 73%–86%
            t=6h   : 80%–91%
            t=10h  : 86%–95%
            t=18h  : 92%–98%
            t=24h  : 100%     (ground truth)

        upvote_ratio: stable với noise nhỏ ±3%, không dùng growth ratio.
        """
        rng = np.random.default_rng(seed)

        if growth_profiles is None:
            growth_profiles = [
                {"delta_h":  0.0, "lo": 0.05, "hi": 0.15},
                {"delta_h":  0.5, "lo": 0.15, "hi": 0.35},
                {"delta_h":  1.0, "lo": 0.30, "hi": 0.55},
                {"delta_h":  1.5, "lo": 0.45, "hi": 0.65},
                {"delta_h":  2.0, "lo": 0.55, "hi": 0.72},
                {"delta_h":  3.0, "lo": 0.65, "hi": 0.80},
                {"delta_h":  4.0, "lo": 0.73, "hi": 0.86},
                {"delta_h":  6.0, "lo": 0.80, "hi": 0.91},
                {"delta_h": 10.0, "lo": 0.86, "hi": 0.95},
                {"delta_h": 18.0, "lo": 0.92, "hi": 0.98},
                {"delta_h": 24.0, "lo": 1.00, "hi": 1.00},
            ]

        col_map: dict[str, str] = {
            "ups":          "score",
            "comments":     "num_comments",
            "comment_count": "num_comments",
            "ratio":        "upvote_ratio",
        }
        df = df.copy()
        for raw, std in col_map.items():
            if raw in df.columns and std not in df.columns:
                df = df.rename(columns={raw: std})

        if "post_id" not in df.columns:
            if "id" in df.columns:
                df["post_id"] = df["id"].astype(str)
            else:
                df["post_id"] = [f"reddit_{i:05d}" for i in range(len(df))]

        # Parse timestamp (Reddit dùng UNIX seconds)
        def _parse_ts(series: pd.Series) -> pd.Series:
            try:
                parsed = pd.to_datetime(series, unit="s", errors="coerce", utc=True)
                if parsed.isna().all():
                    parsed = pd.to_datetime(series, errors="coerce", utc=True)
            except Exception:
                parsed = pd.to_datetime(series, errors="coerce", utc=True)
            if parsed.dt.tz is None:
                parsed = parsed.dt.tz_localize("UTC")
            return parsed

        if crawl_time_col in df.columns:
            base_times = _parse_ts(df[crawl_time_col])
        elif "created_utc" in df.columns:
            base_times = _parse_ts(df["created_utc"])
        else:
            base_times = pd.Series(
                [pd.Timestamp("2026-01-01", tz="UTC")] * len(df),
                index=df.index,
            )
        base_times = base_times.fillna(pd.Timestamp("2026-01-01", tz="UTC"))

        rows: list[dict] = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            base_ts = base_times.iloc[idx]

            final_score    = float(row.get("score",        0))
            final_comments = float(row.get("num_comments", 0))
            final_ratio    = float(row.get("upvote_ratio", 0.7))

            for profile in growth_profiles:
                ratio = rng.uniform(profile["lo"], profile["hi"])
                noise = rng.uniform(0.95, 1.05)

                snap = row.to_dict()
                snap["crawl_time"]    = base_ts + pd.Timedelta(hours=profile["delta_h"])
                snap["score"]         = max(0, round(final_score    * ratio * noise))
                snap["num_comments"]  = max(0, round(final_comments * ratio * noise))
                # upvote_ratio: stable với noise nhỏ
                ratio_noise = rng.uniform(0.97, 1.03)
                snap["upvote_ratio"]  = float(np.clip(final_ratio * ratio_noise, 0.0, 1.0))
                rows.append(snap)

        result = pd.DataFrame(rows).reset_index(drop=True)
        for raw in col_map:
            if raw in result.columns:
                result = result.drop(columns=[raw], errors="ignore")

        return result
