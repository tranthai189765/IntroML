"""
PA Classification Head — platform-agnostic.

PAClassificationHead dự đoán popularity tier {0,1,2,3}.
Label names mặc định là Low/Medium/Popular/Viral nhưng configurable.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import SGDClassifier


LABEL_NAMES = {0: "Low", 1: "Medium", 2: "Popular", 3: "Viral"}


@dataclass
class ClassificationStepResult:
    index: int
    y_pred: int
    y_true: int
    correct: bool
    y_pred_label: str
    y_true_label: str


class PAClassificationHead:
    """
    PA Classifier dự đoán label {0..n_classes-1}.

    Parameters
    ----------
    n_classes : int
        Số lớp phân loại. Mặc định 4 (Low/Medium/Popular/Viral).
    label_names : dict[int, str] | None
        Tên nhãn cho từng lớp. Mặc định = LABEL_NAMES.
    """

    def __init__(
        self,
        C: float = 1.0,
        loss: str = "hinge",
        n_classes: int = 4,
        label_names: dict[int, str] | None = None,
    ):
        lr_mode = "pa2" if loss == "squared_hinge" else "pa1"
        self.model = SGDClassifier(
            loss=loss, penalty=None, learning_rate=lr_mode,
            eta0=C, max_iter=1, tol=None,
            random_state=42, warm_start=False,
        )
        self.n_classes = n_classes
        self._label_names = label_names if label_names is not None else LABEL_NAMES
        self._classes = np.arange(n_classes)
        self.history: list[ClassificationStepResult] = []
        self._initialized = False

    def run_online(
        self, Z: np.ndarray, y_cls: np.ndarray
    ) -> list[ClassificationStepResult]:
        n = len(Z)
        self.history = []

        self.model.partial_fit(Z[:1], y_cls[:1], classes=self._classes)
        self._initialized = True

        for idx in range(1, n):
            x = Z[idx: idx + 1]
            y_true = int(y_cls[idx])
            y_pred = int(self.model.predict(x)[0])

            self.history.append(ClassificationStepResult(
                index=idx,
                y_pred=y_pred, y_true=y_true,
                correct=(y_pred == y_true),
                y_pred_label=self._label_names.get(y_pred, "?"),
                y_true_label=self._label_names.get(y_true, "?"),
            ))

            self.model.partial_fit(x, [y_true])

        return self.history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.model.predict(Z)

    def predict_single(self, z: np.ndarray) -> int:
        return int(self.model.predict(z.reshape(1, -1))[0])

    def predict_proba_approx(self, Z: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(Z)
        return np.zeros((len(Z), self.n_classes))

    def label_name(self, label: int) -> str:
        return self._label_names.get(label, "?")
