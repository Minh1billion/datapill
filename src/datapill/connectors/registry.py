from __future__ import annotations

from dataclasses import fields
from typing import Any

from .base import BaseConnector
from .sources.kafka import KafkaConnector, KafkaConnectorConfig
from .sources.local_directory import LocalDirectoryConnector, LocalConnectorConfig
from .sources.mysql import MySQLConnector, MySQLConnectorConfig
from .sources.postgresql import PostgreSqlConnector, PostgreSQLConnectorConfig
from .sources.rest_api import RestApiConnector, RestApiConnectorConfig
from .sources.s3 import S3Connector, S3ConnectorConfig
from .sources.sqlite import SQLiteConnector, SQLiteConnectorConfig


_CONNECTORS: dict[str, tuple[type[BaseConnector], type]] = {
    "postgres":  (PostgreSqlConnector,      PostgreSQLConnectorConfig),
    "mysql":     (MySQLConnector,           MySQLConnectorConfig),
    "s3":        (S3Connector,              S3ConnectorConfig),
    "kafka":     (KafkaConnector,           KafkaConnectorConfig),
    "local":     (LocalDirectoryConnector,  LocalConnectorConfig),
    "sqlite":    (SQLiteConnector,          SQLiteConnectorConfig),
    "rest":      (RestApiConnector,         RestApiConnectorConfig),
}

_SENSITIVE = {"password", "secret_key", "sasl_password", "auth_token", "basic_password"}


def sources() -> list[str]:
    return list(_CONNECTORS.keys())


def build(source: str, config: dict[str, Any]) -> BaseConnector:
    if source not in _CONNECTORS:
        raise ValueError(f"unknown source: {source!r}. available: {sources()}")
    connector_cls, config_cls = _CONNECTORS[source]
    known = {f.name for f in fields(config_cls)}
    return connector_cls(config_cls(**{k: v for k, v in config.items() if k in known}))


def safe_config(source: str, config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in _SENSITIVE}