"""Catalogue identity constraints shared by discovery, ingestion and replay."""

from typing import Any
from uuid import UUID


def product_id(value: str) -> str:
    """Canonical UUIDs are safe catalogue filters and filesystem identifiers."""
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("Satellite product ID must be a UUID") from error


def require_product(payload: dict[str, Any], expected: str) -> None:
    if product_id(payload.get("sar_product_id", "")) != product_id(expected):
        raise ValueError("Processed satellite product ID differs from selected product")
