# Re-export from pa_core for backwards compatibility.
from ...pa_core.regression_head import (
    QuadRegressionHead,
    SinglePARegressor,
    RegressionStepResult,
    TARGET_NAMES,
    _LOG_CLIP,
)

__all__ = ["QuadRegressionHead", "SinglePARegressor", "RegressionStepResult", "TARGET_NAMES", "_LOG_CLIP"]
