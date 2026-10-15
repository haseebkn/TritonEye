"""Search public corroboration leads before any new SAR or optical interpretation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import shape

from agents.acquisition import native_acquisition_group
from agents.artifacts import REPO_ROOT
from agents.cd_source_refresh import response_status, summarize_optical
from agents.cd_training_reservations import freeze_priors
from agents.hh_hv_reservation import save
from agents.iceberg_corroboration import query_optical
from agents.nl_reception import role
from agents.provisional_training import GUARDS
from agents.source_followup import fresh
from agents.source_reservation import checked, envelope, ref
from agents.training_ais_coverage import SELECTION

POLICY = REPO_ROOT / "configs/reservations/nl_iceberg_search_v010.json"
REPORT = REPO_ROOT / "docs/evidence/nl_iceberg_search_v010.json"
LOCK = REPO_ROOT / "configs/reservations/nl_iceberg_search_v010.lock.json"
RANKING = REPO_ROOT / "data/source_supported/v010_01/corrected_audit/iip_ranked.json"
SATELLITE = re.compile(r"^(S1[A-D]|SN|RS|RCM|SENT|TSX|ENVI)")


def anchors(
    ranking: dict[str, Any],
    frozen: dict[str, Any],
    rules: dict[str, Any],
    normalized: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select source observations with a known SAR overlap, before optical scores."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    old = {(a["stream"], str(a["source_identity"])) for a in frozen["old"]}
    candidates = sorted(
        ranking["ranked"],
        key=lambda r: (
            r["absolute_offset_s"],
            r["search_uncertainty_m"],
            r["product"]["Id"],
            r["observation"]["iceberg_id"],
        ),
    )
    for row in candidates:
        obs, product = row["observation"], row["product"]
        identity = obs["iceberg_id"]
        if (
            obs["timestamp"][:4] < str(rules["minimum_year"])
            or obs["method"] not in {"VIS", "R/V"}
            or SATELLITE.match(obs["source"])
            or obs["size"] not in rules["sizes"]
            or row["absolute_offset_s"] > rules["maximum_sar_offset_s"]
            or "_IW_GRDH_1SDH_" not in product["Name"]
        ):
            counts["outside_source_screen"] += 1
            continue
        group = native_acquisition_group(product["Name"])
        decision = role(
            envelope(obs["lon"], obs["lat"], rules["half_width_m"]),
            frozen,
            group or "unknown",
        )
        if group is None or not decision["split_safe"]:
            counts[decision["exclusion"] or "missing_native_acquisition_group"] += 1
            continue
        if ("iceberg_support", str(identity)) in old:
            counts["previously_reserved_source_identity"] += 1
            continue
        if identity in seen:
            counts["repeated_source_identity"] += 1
            continue
        if (
            groups[decision["geographic_group"]]
            >= rules["maximum_per_geographic_group"]
        ):
            counts["geographic_batch_cap"] += 1
            continue
        if len(selected) >= rules["maximum_anchors"]:
            counts["batch_cap"] += 1
            continue
        selected.append(
            {
                "id": f"iceberg_source_anchor_{len(selected) + 1:02d}",
                "source": obs,
                "product_id": product["Id"],
                "name": product["Name"],
                "native_acquisition_group": group,
                "geographic_group": decision["geographic_group"],
                "source_sar_offset_s": row["absolute_offset_s"],
                "search_uncertainty_m": row["search_uncertainty_m"],
                "label": None,
                "reserved": False,
                **GUARDS,
            }
        )
        seen.add(identity)
        groups[decision["geographic_group"]] += 1
    # If previous SAR-centred searches exhausted the eligible source pool,
    # select new primary observations first. These have no established SAR
    # overlap: optical coverage is screened before further SAR acquisition.
    for obs in sorted(
        (normalized or {}).get("observations", []),
        key=lambda o: (o["timestamp"], o["iceberg_id"]),
        reverse=True,
    ):
        if len(selected) >= rules["maximum_anchors"]:
            break
        identity = obs["iceberg_id"]
        if (
            obs["timestamp"][:4] < str(rules["minimum_year"])
            or obs["method"] not in {"VIS", "R/V"}
            or SATELLITE.match(obs["source"])
            or obs["size"] not in rules["sizes"]
            or identity in seen
            or ("iceberg_support", str(identity)) in old
        ):
            continue
        decision = role(
            envelope(obs["lon"], obs["lat"], rules["half_width_m"]),
            frozen,
            "source_metadata_only",
        )
        if (
            not decision["split_safe"]
            or groups[decision["geographic_group"]]
            >= rules["maximum_per_geographic_group"]
        ):
            continue
        selected.append(
            {
                "id": f"iceberg_source_anchor_{len(selected) + 1:02d}",
                "source": obs,
                "product_id": None,
                "name": None,
                "native_acquisition_group": None,
                "geographic_group": decision["geographic_group"],
                "source_sar_offset_s": None,
                "search_uncertainty_m": None,
                "label": None,
                "reserved": False,
                **GUARDS,
            }
        )
        seen.add(identity)
        groups[decision["geographic_group"]] += 1
        counts["source_first_without_established_sar_overlap"] += 1
    return {"anchors": selected, "screen_counts": dict(counts)}


def requests_for(
    anchor: dict[str, Any], rules: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    obs = anchor["source"]
    instant = datetime.fromisoformat(obs["timestamp"])
    span = timedelta(seconds=rules["optical_half_window_s"])
    start, end = [
        (instant + delta).isoformat().replace("+00:00", "Z") for delta in (-span, span)
    ]
    allowance = (
        rules["position_allowance_m"]
        + rules["drift_discovery_m_per_s"] * rules["optical_half_window_s"]
    )
    geometry = envelope(obs["lon"], obs["lat"], allowance)
    params = {
        "$filter": "Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL1C') "
        + f"and ContentDate/Start ge {start} and ContentDate/Start le {end} "
        + f"and OData.CSC.Intersects(area=geography'SRID=4326;{shape(geometry).wkt}')",
        "$orderby": "ContentDate/Start asc",
        "$top": "100",
        "$expand": "Attributes",
    }
    stac = {
        "collections": ["landsat-c2-l2"],
        "bbox": list(shape(geometry).bounds),
        "datetime": f"{start}/{end}",
        "limit": 100,
    }
    target = {"lon": obs["lon"], "lat": obs["lat"], "native_time_utc": obs["timestamp"]}
    return params, stac, target


def fetch_pages(output: Path, rules: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for i, source in enumerate(rules["public_sources"]):
        path = output / f"public_source_{i:02d}.html"
        receipt: dict[str, Any] = {
            "url": source["url"],
            "purpose": source["purpose"],
            "same_reflector_confirmed": False,
        }
        try:
            response = requests.get(source["url"], timeout=45)
            response.raise_for_status()
            path.write_bytes(response.content)
            receipt.update(
                status="retrieved",
                response=ref(path),
                final_url=response.url,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            )
        except requests.RequestException as error:
            save(path.with_suffix(".json"), {"error_type": type(error).__name__})
            receipt.update(
                status="failed",
                response=ref(path.with_suffix(".json")),
                error_type=type(error).__name__,
            )
        results.append(receipt)
    return results


def search(output: Path) -> None:
    if REPORT.exists() or LOCK.exists():
        raise ValueError("Fresh public evidence version required")
    fresh(output)
    rules = json.loads(POLICY.read_text())
    frozen = freeze_priors(SELECTION)
    ranking = json.loads(RANKING.read_text())
    checked(ranking["provenance"]["normalized"])
    normalized = json.loads(checked(ranking["provenance"]["normalized"]).read_text())
    for source in normalized["sources"]:
        checked(source)
    selected = anchors(ranking, frozen, rules, normalized)
    # Freeze selection before any new optical catalogue response is inspected.
    save(output / "selection.json", selected)
    save(output / "priors.json", frozen)
    pages = fetch_pages(output, rules)
    results = []
    for anchor in selected["anchors"]:
        params, stac, target = requests_for(anchor, rules)
        receipt = query_optical(params, output / f'{anchor["id"]}_s2.json')
        page = json.loads(checked(receipt["response"]).read_text())
        s2 = (
            summarize_optical(page, target, False)
            if receipt["status"] != "failed"
            else []
        )
        path = output / f'{anchor["id"]}_landsat.json'
        try:
            response = requests.post(rules["landsat_endpoint"], json=stac, timeout=45)
            response.raise_for_status()
            payload = response.json()
            status = response_status(payload, True)
            save(path, payload)
            ls = summarize_optical(payload, target, True)
        except (requests.RequestException, ValueError) as error:
            save(path, {"error_type": type(error).__name__})
            status, ls = "failed", []
        results.append(
            {
                "anchor": anchor,
                "sentinel2": {"request": params, "receipt": receipt, "products": s2},
                "landsat": {
                    "request": stac,
                    "response": ref(path),
                    "status": status,
                    "products": ls,
                },
                "pixels_inspected": False,
                "physical_corroboration": False,
            }
        )
        save(output / f"progress_{len(results)}.json", {"results": results})
        print(
            json.dumps(
                {
                    "anchor": anchor["id"],
                    "s2": len(s2),
                    "landsat": len(ls),
                    "statuses": [receipt["status"], status],
                }
            ),
            flush=True,
        )
    private = {"results": results, "pages": pages}
    save(output / "search.json", private)
    publish(output)


def public_summary(output: Path) -> dict[str, Any]:
    rules = json.loads(POLICY.read_text())
    ranking = json.loads(RANKING.read_text())
    selected = json.loads((output / "selection.json").read_text())
    private = json.loads((output / "search.json").read_text())
    return {
        "version": "0.1.0",
        "inputs": {
            "policy": ref(POLICY),
            "ranking": ref(RANKING),
            "normalized": ranking["provenance"]["normalized"],
            "priors": ref(output / "priors.json"),
            "selection": ref(output / "selection.json"),
            "search": ref(output / "search.json"),
            "implementation": ref(Path(__file__)),
        },
        "screen_counts": selected["screen_counts"],
        "anchors": [
            {
                "id": r["anchor"]["id"],
                "product_id": r["anchor"]["product_id"],
                "name": r["anchor"]["name"],
                "geographic_group": r["anchor"]["geographic_group"],
                "source_utc": r["anchor"]["source"]["timestamp"],
                "method": r["anchor"]["source"]["method"],
                "source_sar_offset_s": r["anchor"]["source_sar_offset_s"],
                "sentinel2_status": r["sentinel2"]["receipt"]["status"],
                "landsat_status": r["landsat"]["status"],
                "sentinel2_products": r["sentinel2"]["products"],
                "landsat_products": r["landsat"]["products"],
                "physical_corroboration": False,
                "label": None,
                "reserved": False,
            }
            for r in private["results"]
        ],
        "public_sources": private["pages"],
        "labels_admitted": 0,
        "accuracy_truth_admitted": False,
        "prior_project_exposure_disclosed": True,
        "limits": rules["limits"],
        **GUARDS,
    }


def publish(output: Path) -> None:
    if REPORT.exists() or LOCK.exists():
        raise ValueError("Fresh public evidence version required")
    report = public_summary(output)
    save(REPORT, report)
    save(LOCK, {"report": ref(REPORT)})
    print(json.dumps({"anchors": len(report["anchors"]), "labels_admitted": 0}))


def verify(local: bool = False) -> dict[str, Any]:
    report = json.loads(checked(json.loads(LOCK.read_text())["report"]).read_text())
    rules = json.loads(checked(report["inputs"]["policy"]).read_text())
    checked(report["inputs"]["implementation"])
    if (
        any(report.get(k) is not v for k, v in GUARDS.items())
        or report["labels_admitted"] != 0
        or any(
            a["physical_corroboration"] is not False
            or a["label"] is not None
            or a["reserved"] is not False
            for a in report["anchors"]
        )
    ):
        raise ValueError("Provisional evidence guard changed")
    if local:
        inputs = report["inputs"]
        frozen = json.loads(checked(inputs["priors"]).read_text())
        for path, digest in frozen["prior_metadata_hashes"].items():
            checked({"path": path, "sha256": digest})
        normalized = json.loads(checked(inputs["normalized"]).read_text())
        selected = anchors(
            json.loads(checked(inputs["ranking"]).read_text()),
            frozen,
            rules,
            normalized,
        )
        if (
            selected != json.loads(checked(inputs["selection"]).read_text())
            or selected["screen_counts"] != report["screen_counts"]
        ):
            raise ValueError("Source selection replay differs")
        for source in normalized["sources"]:
            checked(source)
        private = json.loads(checked(inputs["search"]).read_text())
        if public_summary(checked(inputs["search"]).parent) != report:
            raise ValueError("Public source summary replay differs")
        if len(private["results"]) != len(selected["anchors"]) or len(
            report["anchors"]
        ) != len(selected["anchors"]):
            raise ValueError("Search inventory differs")
        for a, r, p in zip(
            selected["anchors"], private["results"], report["anchors"], strict=True
        ):
            params, stac, target = requests_for(a, rules)
            if (
                r["anchor"] != a
                or r["sentinel2"]["request"] != params
                or r["landsat"]["request"] != stac
            ):
                raise ValueError("Optical request/anchor changed")
            for key, is_stac in (("sentinel2", False), ("landsat", True)):
                receipt = r[key] if is_stac else r[key]["receipt"]
                payload = json.loads(checked(receipt["response"]).read_text())
                status = response_status(payload, is_stac)
                expected = (
                    summarize_optical(payload, target, is_stac)
                    if status != "failed"
                    else []
                )
                if (
                    receipt["status"] != status
                    or r[key]["products"] != expected
                    or p[key + "_status"] != status
                    or p[key + "_products"] != expected
                ):
                    raise ValueError("Optical response replay differs")
        if private["pages"] != report["public_sources"]:
            raise ValueError("Public source inventory differs")
        for page in private["pages"]:
            checked(page["response"])
    return {
        "verified": True,
        "local": local,
        "anchors": len(report["anchors"]),
        "labels_admitted": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if sum(bool(v) for v in (args.search, args.publish, args.verify)) != 1:
        parser.error("Choose search, publish or verify")
    if args.search:
        search(args.search)
    elif args.publish:
        publish(args.publish)
    elif args.verify:
        print(json.dumps(verify(args.local)))
