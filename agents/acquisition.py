"""Catalogue identity constraints shared by discovery, ingestion and replay."""

import re
from typing import Any
from uuid import UUID


def native_acquisition_group(name: str | None) -> str | None:
    match = re.fullmatch(
        r"(S1[A-D])_(?:IW|EW|SM|WV|S[1-6])_[A-Z0-9]{4}_[A-Z0-9]{4}_"
        r"\d{8}T\d{6}_\d{8}T\d{6}_(\d{6})_([0-9A-F]{6})_"
        r"[0-9A-F]{4}(?:_COG)?(?:\.SAFE)?",
        name or "",
    )
    return "_".join(match.groups()) if match else None


def product_id(value: str) -> str:
    """Canonical UUIDs are safe catalogue filters and filesystem identifiers."""
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("Satellite product ID must be a UUID") from error


def require_product(payload: dict[str, Any], expected: str) -> None:
    if product_id(payload.get("sar_product_id", "")) != product_id(expected):
        raise ValueError("Processed satellite product ID differs from selected product")
