"""
Twitter-specific EvaluationReporter.

Wrapper mỏng quanh utils.evaluation.EvaluationReporter,
tự động truyền Twitter's compute_popularity_score làm score_fn.
"""

from __future__ import annotations

import numpy as np

from ...utils.evaluation.reporter import EvaluationReporter as _BaseReporter
from ..training.target_builder import compute_popularity_score


def _twitter_score_fn(pred_dict: dict[str, np.ndarray]) -> np.ndarray:
    return compute_popularity_score(
        pred_dict.get("likes",    np.zeros(1)),
        pred_dict.get("comments", np.zeros(1)),
        pred_dict.get("reposts",  np.zeros(1)),
        pred_dict.get("views",    np.zeros(1)),
    )


class EvaluationReporter(_BaseReporter):
    """
    EvaluationReporter cho Twitter PA Pipeline.

    Tự động dùng Twitter popularity score formula:
        Score = log(0.01*V + L + 5*C + 10*R + 1)
    """

    def __init__(
        self,
        reg_history: list,
        cls_history: list,
        k: int = 10,
    ):
        super().__init__(
            reg_history=reg_history,
            cls_history=cls_history,
            score_fn=_twitter_score_fn,
            k=k,
        )
