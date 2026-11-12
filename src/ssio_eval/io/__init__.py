"""File-format adapters."""

from __future__ import annotations

from .datasets import LAYOUTS, discover_sequences, find_gt, find_imu, load_trajectory
from .tables import Table, TableError, read_table, resolve_columns, table_to_trajectory

__all__ = [
    "Table",
    "TableError",
    "read_table",
    "resolve_columns",
    "table_to_trajectory",
    "load_trajectory",
    "find_gt",
    "find_imu",
    "discover_sequences",
    "LAYOUTS",
]