"""Catalogue identity constraints shared by discovery, ingestion and replay."""

import json
import re
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from shapely.errors import GEOSException
from shapely.geometry import shape

from agents.ais_validation import parse_utc


def acquisition_start(product: dict[str, Any]) -> str | None:
    dates = product.get("ContentDate")
    value = dates.get("Start") if isinstance(dates, dict) else None
    return value if isinstance(value, str) and parse_utc(value) is not None else None


def _complete_footprint(footprint: Any) -> bool:
    if not isinstance(footprint, dict) or footprint.get("type") not in (
        "Polygon",
        "MultiPolygon",
    ):
        return False
    try:
        coordinates = footprint["coordinates"]
        polygons = [coordinates] if footprint["type"] == "Polygon" else coordinates
        if not polygons or any(not polygon for polygon in polygons):
            return False
        for polygon in polygons:
            for ring in polygon:
                if len(ring) < 4 or ring[0] != ring[-1]:
                    return False
        geometry = shape(footprint)
        return bool(not geometry.is_empty and geometry.is_valid)
    except (GEOSException, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


def unique_acquisitions(products: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count physical slices once, without substituting their selected UUID.

    Original/COG and reprocessed catalogue variants have different CRCs/UUIDs.
    Collapse only recognized native slice identities with equal complete UTC
    intervals and footprint metadata. Unknown or conflicting metadata remains
    separate. Prefer original SAFE, then UUID, independently of page order.
    """
    by_uuid = {product_id(p["Id"]): p for p in products}
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for identity, p in by_uuid.items():
        match = re.fullmatch(
            r"(S1[A-D]_(?:IW|EW|SM|WV|S[1-6])_[A-Z0-9]{4}_[A-Z0-9]{4}_"
            r"\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6})_"
            r"[0-9A-F]{4}(?:_COG)?(?:\.SAFE)?",
            p.get("Name", ""),
        )
        dates = p.get("ContentDate") or {}
        start, end = parse_utc(dates.get("Start")), parse_utc(dates.get("End"))
        footprint = p.get("GeoFootprint")
        key: tuple[str, ...] = ("uuid", identity)
        if match and start and end and start < end and _complete_footprint(footprint):
            key = (
                "slice",
                match.group(1),
                start.isoformat(),
                end.isoformat(),
                json.dumps(footprint, sort_keys=True),
            )
        groups.setdefault(key, []).append(p)
    selected = []
    for variants in groups.values():
        variants.sort(key=lambda p: ("_COG" in p["Name"], product_id(p["Id"])))
        choice = dict(variants[0])
        if len(variants) > 1:
            choice["catalogue_aliases"] = [
                {"Id": product_id(p["Id"]), "Name": p["Name"]} for p in variants
            ]
        selected.append(choice)
    return sorted(
        selected,
        key=lambda p: (acquisition_start(p) or "", product_id(p["Id"])),
        reverse=True,
    )


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
