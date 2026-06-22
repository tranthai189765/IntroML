# Re-export from pa_core for backwards compatibility.
from ...pa_core.classification_head import (
    PAClassificationHead,
    ClassificationStepResult,
    LABEL_NAMES,
)

__all__ = ["PAClassificationHead", "ClassificationStepResult", "LABEL_NAMES"]
