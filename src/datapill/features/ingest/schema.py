from dataclasses import dataclass, field
from typing import AsyncGenerator, Any

from datapill.connectors.base import BaseConnector


@dataclass
class IngestConfig:
    connector: BaseConnector
    query: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)