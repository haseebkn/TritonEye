#!/usr/bin/env python3
"""
TritonEye Sentinel-1 Radiometric Calibration

Converts Level-1 GRD digital numbers to calibrated sigma-nought (σ⁰) in
decibels, using the sigmaNought lookup table shipped in each product's
`annotation/calibration/` XML.

Raw GRD DN is an uncalibrated integer whose absolute value depends on the
processor's scaling, not on how much energy the surface actually scattered
back. Anything comparing pixels across scenes, sensors or incidence angles —
and any detector trained on calibrated imagery — needs σ⁰ instead.

    σ⁰ = DN² / A²        where A is the interpolated sigmaNought LUT value
    σ⁰_dB = 10·log₁₀(σ⁰)

The LUT is a coarse grid (typically ~27 azimuth lines × ~648 range pixels for
an IW GRD scene) that is bilinearly interpolated up to full raster resolution.
"""

import os
import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Tuple

import numpy as np

FloatArray = np.ndarray[Any, Any]

# σ⁰ floor before the log, so nodata and zero-backscatter pixels produce a very
# negative dB value instead of -inf.
_MIN_SIGMA0 = 1e-10

# dB assigned to pixels with no data (DN == 0). Well below any real ocean or
# land return, so downstream normalisation drives them to zero rather than
# treating them as dark water.
NODATA_DB = -60.0


class CalibrationError(RuntimeError):
    """Raised when a product has no usable calibration LUT."""


def find_calibration_xml(measurement_path: str) -> Optional[str]:
    """
    Locates the calibration XML matching a measurement TIFF.

    Products lay this out as
        <product>.SAFE/measurement/s1a-iw-grd-vv-....tiff
        <product>.SAFE/annotation/calibration/calibration-s1a-iw-grd-vv-....xml
    so the polarisation is recovered from the measurement filename and used to
    pick the right LUT — calibrating VH with the VV table would introduce a
    silent multi-dB bias.
    """
    measurement_dir = os.path.dirname(os.path.abspath(measurement_path))
    safe_dir = os.path.dirname(measurement_dir)
    cal_dir = os.path.join(safe_dir, "annotation", "calibration")
    if not os.path.isdir(cal_dir):
        return None

    name = os.path.basename(measurement_path).lower()
    pol = None
    for candidate in ("vv", "vh", "hh", "hv"):
        if f"-{candidate}-" in name or name.endswith(f"{candidate}.tiff"):
            pol = candidate
            break

    files = [f for f in os.listdir(cal_dir) if f.startswith("calibration-")]
    if pol:
        for f in files:
            if f"-{pol}-" in f.lower():
                return os.path.join(cal_dir, f)
    return os.path.join(cal_dir, files[0]) if files else None


def read_sigma_nought_lut(
    xml_path: str,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Parses a calibration XML into (lines, pixels, sigma_nought).

    Returns the LUT's azimuth line indices, its range pixel indices, and the
    sigmaNought values as a (n_lines, n_pixels) grid.
    """
    root = ET.parse(xml_path).getroot()
    vectors = root.findall(".//calibrationVector")
    if not vectors:
        raise CalibrationError(f"No calibrationVector entries in {xml_path}")

    lines: List[float] = []
    rows: List[List[float]] = []
    pixels: Optional[List[float]] = None

    for vec in vectors:
        line_text = vec.findtext("line")
        pixel_text = vec.findtext("pixel")
        sigma_text = vec.findtext("sigmaNought")
        if line_text is None or pixel_text is None or sigma_text is None:
            continue
        pix = [float(v) for v in pixel_text.split()]
        sig = [float(v) for v in sigma_text.split()]
        if len(pix) != len(sig):
            continue
        if pixels is None:
            pixels = pix
        elif len(pix) != len(pixels):
            # Ragged LUT rows would break the regular-grid assumption below.
            continue
        lines.append(float(line_text))
        rows.append(sig)

    if pixels is None or not rows:
        raise CalibrationError(f"Unusable calibration LUT in {xml_path}")

    return np.asarray(lines), np.asarray(pixels), np.asarray(rows)


def interpolate_lut(
    lut_lines: FloatArray,
    lut_pixels: FloatArray,
    lut_values: FloatArray,
    rows: FloatArray,
    cols: FloatArray,
) -> FloatArray:
    """
    Bilinearly interpolates the coarse LUT onto arbitrary (row, col) positions.

    Implemented directly rather than via scipy so calibration adds no dependency
    beyond numpy. Positions outside the LUT are clamped to its edge, which
    matters because the LUT's last pixel index can fall a few columns short of
    the raster width.
    """
    r = np.clip(rows, lut_lines[0], lut_lines[-1])
    c = np.clip(cols, lut_pixels[0], lut_pixels[-1])

    i = np.clip(np.searchsorted(lut_lines, r, side="right") - 1, 0, len(lut_lines) - 2)
    j = np.clip(
        np.searchsorted(lut_pixels, c, side="right") - 1, 0, len(lut_pixels) - 2
    )

    r0, r1 = lut_lines[i], lut_lines[i + 1]
    c0, c1 = lut_pixels[j], lut_pixels[j + 1]

    dr = np.where(r1 > r0, (r - r0) / np.where(r1 > r0, r1 - r0, 1.0), 0.0)
    dc = np.where(c1 > c0, (c - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)

    v00 = lut_values[i, j]
    v01 = lut_values[i, j + 1]
    v10 = lut_values[i + 1, j]
    v11 = lut_values[i + 1, j + 1]

    top = v00 * (1.0 - dc) + v01 * dc
    bottom = v10 * (1.0 - dc) + v11 * dc
    return np.asarray(top * (1.0 - dr) + bottom * dr)


def interpolate_lut_grid(
    lut_lines: FloatArray,
    lut_pixels: FloatArray,
    lut_values: FloatArray,
    rows: FloatArray,
    cols: FloatArray,
) -> FloatArray:
    """
    Interpolates the LUT onto a full (len(rows), len(cols)) grid.

    Takes 1-D row and column indices rather than the broadcast 2-D coordinate
    grids `interpolate_lut` expects. Bilinear interpolation on a regular grid is
    separable, so the row and column weights are computed once along each axis
    and combined, instead of evaluating every intermediate array at full raster
    size. That matters: a 2048x2048 tile through the pointwise path allocates
    well over a dozen 32 MB float64 temporaries and can exhaust host memory
    partway through a scene.

    Interpolation runs in float32 — a calibration LUT holds ~3 significant
    figures, so the extra precision buys nothing here and doubles the footprint.

    That reasoning does NOT extend to the dB conversion in `to_sigma0_db`.
    float32 through the log costs ~1.5e-6 dB, which exceeds this module's own
    accuracy test; the claim that "the result feeds a log so precision does not
    matter" was measured and found false. See the note at that call site.
    """
    r = np.clip(rows, lut_lines[0], lut_lines[-1])
    c = np.clip(cols, lut_pixels[0], lut_pixels[-1])

    i = np.clip(np.searchsorted(lut_lines, r, side="right") - 1, 0, len(lut_lines) - 2)
    j = np.clip(
        np.searchsorted(lut_pixels, c, side="right") - 1, 0, len(lut_pixels) - 2
    )

    r0, r1 = lut_lines[i], lut_lines[i + 1]
    c0, c1 = lut_pixels[j], lut_pixels[j + 1]
    dr = np.where(r1 > r0, (r - r0) / np.where(r1 > r0, r1 - r0, 1.0), 0.0)
    dc = np.where(c1 > c0, (c - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
    dr = dr.astype("float32")[:, None]
    dc = dc.astype("float32")[None, :]

    lv = lut_values.astype("float32", copy=False)
    v00 = lv[np.ix_(i, j)]
    v01 = lv[np.ix_(i, j + 1)]
    v10 = lv[np.ix_(i + 1, j)]
    v11 = lv[np.ix_(i + 1, j + 1)]

    # Interpolate along columns, then rows, writing in place so the four corner
    # arrays are the only full-size allocations.
    np.subtract(v01, v00, out=v01)
    np.multiply(v01, dc, out=v01)
    np.add(v00, v01, out=v00)  # v00 now holds the top edge

    np.subtract(v11, v10, out=v11)
    np.multiply(v11, dc, out=v11)
    np.add(v10, v11, out=v10)  # v10 now holds the bottom edge

    np.subtract(v10, v00, out=v10)
    np.multiply(v10, dr, out=v10)
    np.add(v00, v10, out=v00)
    return np.asarray(v00)


class Calibrator:
    """
    Applies a product's sigmaNought LUT to windows of its measurement raster.

    Build one per band and reuse it; parsing the XML is far more expensive than
    calibrating a tile.
    """

    def __init__(self, xml_path: str) -> None:
        self.xml_path = xml_path
        self.lut_lines, self.lut_pixels, self.lut_values = read_sigma_nought_lut(
            xml_path
        )

    @classmethod
    def for_measurement(cls, measurement_path: str) -> Optional["Calibrator"]:
        """Returns a Calibrator for a measurement TIFF, or None if no LUT exists."""
        xml_path = find_calibration_xml(measurement_path)
        if not xml_path or not os.path.exists(xml_path):
            return None
        try:
            return cls(xml_path)
        except CalibrationError:
            return None

    def to_sigma0_db(
        self, dn: FloatArray, row_offset: int = 0, col_offset: int = 0
    ) -> FloatArray:
        """
        Converts a DN window to σ⁰ in dB.

        row_offset/col_offset give the window's position in the full raster, so
        the correct part of the LUT is sampled — the LUT varies strongly across
        range, and calibrating a tile as if it sat at the scene origin would
        bias it by several dB.
        """
        dn = np.asarray(dn, dtype="float32")
        h, w = dn.shape
        rows = np.arange(row_offset, row_offset + h, dtype="float64")
        cols = np.arange(col_offset, col_offset + w, dtype="float64")

        # Bilinear interpolation on a regular grid is separable, so this is
        # evaluated once per axis rather than per pixel. Doing it pointwise on a
        # full tile allocates enough float64 temporaries to exhaust host memory
        # midway through a scene.
        a = interpolate_lut_grid(
            self.lut_lines, self.lut_pixels, self.lut_values, rows, cols
        )

        # The LUT interpolation runs in float32 -- that is where the memory
        # went, and a calibration LUT carries ~3 significant figures anyway.
        # The dB conversion does NOT: log10 in float32 costs ~1.5e-6 dB, which
        # exceeded this module's own accuracy test and passed locally only on a
        # numpy version that happened to round favourably. Promoting to float64
        # for the ratio and the log keeps one full-size array rather than the
        # dozen-plus the original pointwise path allocated, so the memory win
        # survives while the output stays exact.
        valid = (a > 0) & (dn > 0)
        dn64 = dn.astype("float64")
        a64 = a.astype("float64")
        # Reuse two work arrays: separate squared operands plus a new ratio
        # created several extra 32 MiB buffers per 2048-pixel tile.
        np.square(dn64, out=dn64)
        np.square(a64, out=a64)
        np.divide(dn64, a64, out=dn64, where=valid)
        dn64[~valid] = 0
        sigma0 = dn64
        np.maximum(sigma0, _MIN_SIGMA0, out=sigma0)
        db = np.log10(sigma0, out=sigma0)
        np.multiply(db, 10.0, out=db)
        db[dn <= 0] = NODATA_DB
        return np.asarray(db, dtype="float64")
