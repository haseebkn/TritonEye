import glob
import math
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.calibration import (
    NODATA_DB,
    CalibrationError,
    Calibrator,
    find_calibration_xml,
    interpolate_lut,
    read_sigma_nought_lut,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def write_cal_xml(
    path: Path, lines: List[int], pixels: List[int], value: float
) -> None:
    """A calibration XML with a constant sigmaNought, so σ⁰ is analytic."""
    vectors = []
    for ln in lines:
        vectors.append(
            f"<calibrationVector>"
            f"<line>{ln}</line>"
            f"<pixel>{' '.join(str(p) for p in pixels)}</pixel>"
            f"<sigmaNought>{' '.join(f'{value:.6e}' for _ in pixels)}</sigmaNought>"
            f"</calibrationVector>"
        )
    path.write_text(
        "<calibration><calibrationVectorList count='"
        f"{len(lines)}'>{''.join(vectors)}</calibrationVectorList></calibration>",
        encoding="utf-8",
    )


# --------------------------------------------------------------- LUT parsing


def test_reads_lut_grid(tmp_path: Path) -> None:
    p = tmp_path / "calibration-s1a-iw-grd-vv-x.xml"
    write_cal_xml(p, [0, 500, 1000], [0, 100, 200, 300], 500.0)
    lines, pixels, values = read_sigma_nought_lut(str(p))
    assert list(lines) == [0, 500, 1000]
    assert list(pixels) == [0, 100, 200, 300]
    assert values.shape == (3, 4)
    assert np.allclose(values, 500.0)


def test_empty_lut_raises(tmp_path: Path) -> None:
    p = tmp_path / "calibration-empty.xml"
    p.write_text(
        "<calibration><calibrationVectorList/></calibration>", encoding="utf-8"
    )
    with pytest.raises(CalibrationError):
        read_sigma_nought_lut(str(p))


# ------------------------------------------------------------ interpolation


def test_interpolation_reproduces_control_points() -> None:
    lines = np.array([0.0, 100.0])
    pixels = np.array([0.0, 100.0])
    values = np.array([[10.0, 20.0], [30.0, 40.0]])
    rows = np.array([[0.0, 0.0], [100.0, 100.0]])
    cols = np.array([[0.0, 100.0], [0.0, 100.0]])
    out = interpolate_lut(lines, pixels, values, rows, cols)
    assert np.allclose(out, values)


def test_interpolation_midpoint_is_bilinear() -> None:
    lines = np.array([0.0, 100.0])
    pixels = np.array([0.0, 100.0])
    values = np.array([[0.0, 10.0], [20.0, 30.0]])
    out = interpolate_lut(lines, pixels, values, np.array([[50.0]]), np.array([[50.0]]))
    assert out[0, 0] == pytest.approx(15.0)


def test_interpolation_clamps_outside_the_lut() -> None:
    # The LUT's last pixel index can fall short of the raster width, so
    # positions beyond it must clamp rather than extrapolate or wrap.
    lines = np.array([0.0, 100.0])
    pixels = np.array([0.0, 100.0])
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = interpolate_lut(
        lines, pixels, values, np.array([[999.0]]), np.array([[999.0]])
    )
    assert out[0, 0] == pytest.approx(4.0)


# -------------------------------------------------------------- calibration


def test_sigma0_matches_closed_form(tmp_path: Path) -> None:
    # With a constant LUT, σ⁰ = DN²/A² exactly.
    a = 500.0
    p = tmp_path / "calibration-s1a-iw-grd-vv-x.xml"
    write_cal_xml(p, [0, 1000], [0, 1000], a)
    cal = Calibrator(str(p))

    dn = np.array([[100.0, 500.0], [1000.0, 250.0]])
    db = cal.to_sigma0_db(dn)
    for got, raw in zip(db.ravel(), dn.ravel()):
        assert got == pytest.approx(10.0 * math.log10((raw * raw) / (a * a)), abs=1e-6)


def test_nodata_becomes_floor_not_negative_infinity(tmp_path: Path) -> None:
    p = tmp_path / "calibration-s1a-iw-grd-vv-x.xml"
    write_cal_xml(p, [0, 1000], [0, 1000], 500.0)
    cal = Calibrator(str(p))
    db = cal.to_sigma0_db(np.array([[0.0, 100.0]]))
    assert db[0, 0] == NODATA_DB
    assert np.isfinite(db).all()


def test_window_offset_samples_the_right_lut_region(tmp_path: Path) -> None:
    # sigmaNought varies strongly across range; calibrating a tile as if it sat
    # at the scene origin would bias it by several dB.
    p = tmp_path / "calibration-s1a-iw-grd-vv-x.xml"
    vectors = (
        "<calibrationVector><line>0</line><pixel>0 1000</pixel>"
        "<sigmaNought>1.000000e+02 1.000000e+03</sigmaNought></calibrationVector>"
        "<calibrationVector><line>1000</line><pixel>0 1000</pixel>"
        "<sigmaNought>1.000000e+02 1.000000e+03</sigmaNought></calibrationVector>"
    )
    p.write_text(
        f"<calibration><calibrationVectorList count='2'>{vectors}"
        "</calibrationVectorList></calibration>",
        encoding="utf-8",
    )
    cal = Calibrator(str(p))
    dn = np.array([[500.0]])
    at_origin = cal.to_sigma0_db(dn, row_offset=0, col_offset=0)[0, 0]
    far_range = cal.to_sigma0_db(dn, row_offset=0, col_offset=1000)[0, 0]
    assert at_origin > far_range
    assert at_origin - far_range == pytest.approx(20.0, abs=0.5)


def test_missing_calibration_returns_none(tmp_path: Path) -> None:
    fake = tmp_path / "measurement" / "s1a-iw-grd-vv-x.tiff"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"")
    assert Calibrator.for_measurement(str(fake)) is None


# --------------------------------------- real products, when the archive exists


def real_products() -> List[str]:
    pattern = os.path.join(
        REPO_ROOT, "data", "raw", "S1*.SAFE", "measurement", "*vv*.tif*"
    )
    return [p for p in sorted(glob.glob(pattern)) if find_calibration_xml(p)]


@pytest.mark.skipif(
    not real_products(), reason="no real Sentinel-1 products in data/raw"
)
def test_real_products_calibrate_into_physical_range() -> None:
    import rasterio
    from rasterio.windows import Window

    for path in real_products()[:3]:
        cal = Calibrator.for_measurement(path)
        assert cal is not None, path
        with rasterio.open(path) as src:
            r0 = max(0, src.height // 2 - 500)
            c0 = max(0, src.width // 2 - 500)
            dn = src.read(1, window=Window(c0, r0, 1000, 1000)).astype("float64")
        db = cal.to_sigma0_db(dn, row_offset=r0, col_offset=c0)
        valid = db[dn > 0]
        if valid.size < 1000:
            continue
        median = float(np.median(valid))
        # Open-ocean and coastal σ⁰ sits well inside this envelope; a median
        # outside it means the LUT was misapplied or DN was left uncalibrated.
        assert -35.0 < median < 5.0, f"{path}: median {median:.1f} dB"


def test_grid_path_matches_pointwise_path(tmp_path: Path) -> None:
    """
    The separable grid path is an optimisation, not a different algorithm --
    it must agree with the pointwise implementation it replaced.
    """
    from agents.calibration import interpolate_lut_grid

    p = tmp_path / "calibration-s1a-iw-grd-vv-x.xml"
    vectors = "".join(
        f"<calibrationVector><line>{ln}</line>"
        f"<pixel>0 500 1000</pixel>"
        f"<sigmaNought>{100+ln:.6e} {300+ln:.6e} {700+ln:.6e}</sigmaNought>"
        f"</calibrationVector>"
        for ln in (0, 400, 800)
    )
    p.write_text(
        f"<calibration><calibrationVectorList count='3'>{vectors}"
        "</calibrationVectorList></calibration>",
        encoding="utf-8",
    )
    lines, pixels, values = read_sigma_nought_lut(str(p))

    rows = np.linspace(0, 800, 37)
    cols = np.linspace(0, 1000, 53)
    grid = interpolate_lut_grid(lines, pixels, values, rows, cols)

    pointwise = interpolate_lut(
        lines,
        pixels,
        values,
        rows[:, None] * np.ones((1, len(cols))),
        np.ones((len(rows), 1)) * cols[None, :],
    )
    assert grid.shape == pointwise.shape == (37, 53)
    assert np.allclose(grid, pointwise, rtol=1e-5)


def test_large_tile_calibrates_without_excessive_memory(tmp_path: Path) -> None:
    """
    Regression: a full 2048x2048 tile through the old pointwise path allocated
    a dozen-plus 32 MB float64 temporaries and exhausted host memory partway
    through a scene. The grid path must handle a real tile size comfortably.
    """
    p = tmp_path / "calibration-s1a-iw-grd-vv-x.xml"
    write_cal_xml(p, [0, 8000, 16000], [0, 12000, 25000], 600.0)
    cal = Calibrator(str(p))

    dn = np.full((2048, 2048), 200.0)
    db = cal.to_sigma0_db(dn, row_offset=4000, col_offset=6000)
    assert db.shape == (2048, 2048)
    assert np.isfinite(db).all()
    # constant LUT -> closed form
    assert float(np.median(db)) == pytest.approx(
        10.0 * math.log10((200.0 / 600.0) ** 2), abs=1e-3
    )
