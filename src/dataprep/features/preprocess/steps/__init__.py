from ..schema import StepConfig
from .base import BaseStep
from .custom import CustomPython
from .encoding import OneHotEncoder, OrdinalEncoder
from .missing import DropMissing, ImputeMean, ImputeMedian, ImputeMode
from .outlier import ClipIQR, ClipZScore
from .scaling import MinMaxScaler, RobustScaler, StandardScaler
from .structural import (
    CastDtype,
    Deduplicate,
    DropColumns,
    RenameColumns,
    SelectColumns,
)

_REGISTRY: dict[str, type[BaseStep]] = {
    "impute_mean": ImputeMean,
    "impute_median": ImputeMedian,
    "impute_mode": ImputeMode,
    "drop_missing": DropMissing,
    "clip_iqr": ClipIQR,
    "clip_zscore": ClipZScore,
    "standard_scaler": StandardScaler,
    "minmax_scaler": MinMaxScaler,
    "robust_scaler": RobustScaler,
    "onehot": OneHotEncoder,
    "ordinal": OrdinalEncoder,
    "select_columns": SelectColumns,
    "drop_columns": DropColumns,
    "rename_columns": RenameColumns,
    "cast_dtype": CastDtype,
    "deduplicate": Deduplicate,
    "custom_python": CustomPython,
}


def build_step(config: StepConfig) -> BaseStep:
    cls = _REGISTRY.get(config.step)
    if cls is None:
        raise ValueError(f"Unknown step: '{config.step}'. Available: {sorted(_REGISTRY)}")
    return cls(config)