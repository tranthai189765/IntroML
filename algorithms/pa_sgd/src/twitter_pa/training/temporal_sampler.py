"""
Twitter-specific TemporalSampler.

Kế thừa utils.TemporalSampler (create_pairs, _validate, summary).
Thêm simulate_snapshots() với growth profile Twitter viral curve (11 checkpoints).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...utils.temporal_sampler import TemporalSampler as _BaseTemporalSampler


class TemporalSampler(_BaseTemporalSampler):
    """
    Twitter TemporalSampler.

    ENGAGEMENT_COLS = ["likes", "comments", "reposts", "views"]

    Thêm phương thức simulate_snapshots() để sinh multi-snapshot từ single-snapshot CSV.
    """

    ENGAGEMENT_COLS = ["likes", "comments", "reposts", "views"]

    @staticmethod
    def simulate_snapshots(
        df: pd.DataFrame,
        crawl_time_col: str = "crawl_time",
        growth_profiles: list[dict] | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Từ DataFrame 1-snapshot/post, sinh ra multi-snapshot giả lập
        theo viral growth curve Twitter (11 checkpoints).

        Checkpoints: 0h, 0.5h, 1h, 1.5h, 2h, 3h, 4h, 6h, 10h, 18h, 24h.
        t=24h = ground truth (giá trị thực từ CSV).

        Parameters
        ----------
        df : DataFrame
            Phải có cột engagement (likes/comments/reposts/views hoặc
            likeCount/replyCount/retweetCount/viewCount).
        crawl_time_col : str
            Tên cột chứa timestamp gốc.
        growth_profiles : list[dict], optional
            Override growth profile. Mỗi dict cần: delta_h, lo, hi.
        seed : int

        Returns
        -------
        DataFrame với nhiều rows/post, sẵn sàng cho create_pairs().
        """
        rng = np.random.default_rng(seed)

        # Use 8 checkpoints: 0h, 0.5h, 1h, 1.5h, 2h, 3h, 4h, 6h
        if growth_profiles is None:
            growth_profiles = [
                {"delta_h": 0.0, "lo": 0.01, "hi": 0.04},
                {"delta_h": 0.5, "lo": 0.06, "hi": 0.15},
                {"delta_h": 1.0, "lo": 0.12, "hi": 0.28},
                {"delta_h": 1.5, "lo": 0.20, "hi": 0.38},
                {"delta_h": 2.0, "lo": 0.28, "hi": 0.48},
                {"delta_h": 3.0, "lo": 0.38, "hi": 0.58},
                {"delta_h": 4.0, "lo": 0.48, "hi": 0.67},
                {"delta_h": 6.0, "lo": 1.00, "hi": 1.00},
            ]

        col_map = {
            "likeCount":    "likes",
            "replyCount":   "comments",
            "retweetCount": "reposts",
            "viewCount":    "views",
        }
        df = df.copy()
        for raw, std in col_map.items():
            if raw in df.columns and std not in df.columns:
                df = df.rename(columns={raw: std})

        if "post_id" not in df.columns:
            if "id" in df.columns:
                df["post_id"] = df["id"].astype(str)
            else:
                df["post_id"] = [f"post_{i:05d}" for i in range(len(df))]

        def _parse_ts(series: pd.Series) -> pd.Series:
            # Twitter createdAt: "Mon Jun 08 11:19:00 +0000 2026"
            parsed = pd.to_datetime(series, format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
            mask = parsed.isna()
            if mask.any():
                parsed[mask] = pd.to_datetime(series[mask], errors="coerce", utc=True)
            if parsed.dt.tz is None:
                parsed = parsed.dt.tz_localize("UTC")
            else:
                parsed = parsed.dt.tz_convert("UTC")
            return parsed

        if crawl_time_col in df.columns:
            base_times = _parse_ts(df[crawl_time_col])
        elif "createdAt" in df.columns:
            base_times = _parse_ts(df["createdAt"])
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

            final = {
                "likes":    float(row.get("likes",    0)),
                "comments": float(row.get("comments", 0)),
                "reposts":  float(row.get("reposts",  0)),
                "views":    float(row.get("views",    0)),
            }

            for profile in growth_profiles:
                ratio = rng.uniform(profile["lo"], profile["hi"])
                noise = rng.uniform(0.95, 1.05)

                snap = row.to_dict()
                snap["crawl_time"] = base_ts + pd.Timedelta(hours=profile["delta_h"])
                snap["likes"]      = max(0, round(final["likes"]    * ratio * noise))
                snap["comments"]   = max(0, round(final["comments"] * ratio * noise))
                snap["reposts"]    = max(0, round(final["reposts"]  * ratio * noise))
                snap["views"]      = max(0, round(final["views"]    * ratio * noise))
                rows.append(snap)

        result = pd.DataFrame(rows).reset_index(drop=True)
        for raw in col_map:
            if raw in result.columns:
                result = result.drop(columns=[raw], errors="ignore")

        return result
