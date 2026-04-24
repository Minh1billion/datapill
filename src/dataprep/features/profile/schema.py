from typing import Any, Optional
from pydantic import BaseModel, Field


class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class ValueCount(BaseModel):
    value: Any
    count: int
    pct: float


class PatternMatch(BaseModel):
    pattern: str
    count: int
    pct: float


class ColumnProfile(BaseModel):
    name: str
    dtype_physical: str
    dtype_inferred: str
    null_count: int
    null_pct: float
    distinct_count: int
    distinct_pct: float
    is_unique: bool
    min: Optional[Any] = None
    max: Optional[Any] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    percentiles: Optional[dict[str, float]] = None
    histogram: Optional[list[HistogramBin]] = None
    top_values: Optional[list[ValueCount]] = None
    pattern_matches: Optional[list[PatternMatch]] = None
    sample_values: Optional[list[Any]] = None
    warnings: list[str] = Field(default_factory=list)


class DatasetMeta(BaseModel):
    row_count: int
    column_count: int
    memory_mb: float
    sampled: bool
    sample_strategy: str
    sample_size: Optional[int]
    schema_hash: str
    created_at: str


class CorrelationPair(BaseModel):
    col_a: str
    col_b: str
    method: str
    value: float


class DatasetWarning(BaseModel):
    type: str
    column: Optional[str] = None
    message: str


class ProfileDetail(BaseModel):
    profile_id: str
    dataset: DatasetMeta
    columns: list[ColumnProfile]
    correlations: list[CorrelationPair] = Field(default_factory=list)
    warnings: list[DatasetWarning] = Field(default_factory=list)


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    null_pct: float
    distinct_count: int
    min: Optional[Any] = None
    max: Optional[Any] = None
    mean: Optional[float] = None
    top_3_values: Optional[list[Any]] = None
    warnings: list[str] = Field(default_factory=list)


class ProfileSummary(BaseModel):
    profile_id: str
    dataset: dict[str, Any]
    columns: list[ColumnSummary]
    high_level_warnings: list[str] = Field(default_factory=list)