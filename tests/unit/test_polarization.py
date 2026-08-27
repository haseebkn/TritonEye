import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.ingest.ingest_agent import select_polarization_bands

# Real filenames from products in this project's archive.
SDV = [
    "s1d-iw-grd-vh-20260817t212209-20260817t212234-004171-007a29-002.tiff",
    "s1d-iw-grd-vv-20260817t212209-20260817t212234-004171-007a29-001.tiff",
]
SDH = [
    "s1c-iw-grd-hh-20260820t094901-20260820t094926-009077-012049-001-cog.tiff",
    "s1c-iw-grd-hv-20260820t094901-20260820t094926-009077-012049-002-cog.tiff",
]


def test_dual_vv_vh_is_paired_co_then_cross() -> None:
    co, cross, pol = select_polarization_bands(SDV)
    assert pol == "VV/VH"
    assert co is not None and cross is not None
    assert "-vv-" in co
    assert "-vh-" in cross


def test_hh_hv_is_identified_not_relabelled_as_vv() -> None:
    # Regression: a 1SDH product used to fall through to "take the first two
    # TIFFs in directory order", which reported HH as VV and fed the detectors
    # a different scattering regime with no indication anything was wrong.
    co, cross, pol = select_polarization_bands(SDH)
    assert pol == "HH/HV"
    assert co is not None and cross is not None
    assert "-hh-" in co
    assert "-hv-" in cross


def test_directory_order_does_not_decide_the_pairing() -> None:
    # Both orderings must give the same co-pol/cross-pol assignment.
    assert select_polarization_bands(SDV) == select_polarization_bands(SDV[::-1])
    assert select_polarization_bands(SDH) == select_polarization_bands(SDH[::-1])


def test_single_pol_is_refused_rather_than_guessed() -> None:
    co, cross, pol = select_polarization_bands(["s1a-iw-grd-vv-only.tiff"])
    assert (co, cross, pol) == (None, None, "unknown")


def test_unrecognised_naming_is_refused() -> None:
    co, cross, pol = select_polarization_bands(["band_a.tiff", "band_b.tiff"])
    assert (co, cross, pol) == (None, None, "unknown")


def test_short_suffix_naming_is_matched() -> None:
    # The synthetic mock products use bare "...vv.tiff" / "...vh.tiff" names.
    co, cross, pol = select_polarization_bands(
        ["s1a-iw-grd-vh.tiff", "s1a-iw-grd-vv.tiff"]
    )
    assert pol == "VV/VH"
    assert co is not None and cross is not None
    assert co.endswith("vv.tiff")
    assert cross.endswith("vh.tiff")
