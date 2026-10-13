"""Audit and classify authorized Statoil HH/HV training chips, research only."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta

from agents.artifacts import REPO_ROOT, sha256_file, write_json

DEFAULT_INPUT = (
    REPO_ROOT
    / "statoil-iceberg-classifier-challenge/train.json/data/processed/train.json"
)
SEEDS = (20261013, 20261014, 20261015)
SPLIT_SEED = 20261013
NEAR_CORRELATION = 0.995
NEAR_RMSE_DB = 0.5


def available_commit() -> int:
    """Windows commit headroom can be low even when physical RAM is free."""
    if os.name != "nt":
        return int(psutil.virtual_memory().available)

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong)
            for name in (
                "total_physical",
                "available_physical",
                "total_commit",
                "available_commit",
                "total_virtual",
                "available_virtual",
                "extended",
            )
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    windll: Any = getattr(ctypes, "windll")
    if not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot establish memory headroom")
    return int(status.available_commit)


def resource_check() -> None:
    if (
        available_commit() < 3 * 1024**3
        or psutil.virtual_memory().available < 2 * 1024**3
    ):
        raise RuntimeError(
            "Insufficient memory headroom; close other workloads before research"
        )
    for path in (REPO_ROOT, Path(tempfile.gettempdir())):
        if shutil.disk_usage(path).free < 1024**3:
            raise RuntimeError(
                "At least 1 GiB free project and temporary storage required"
            )


def load_training(path: Path) -> tuple[list[str], Any, Any, list[float | None]]:
    """Only labelled training JSON: never open competition or NL test imagery."""
    if path.name != "train.json":
        raise ValueError("Only labelled train.json is permitted")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Nonempty training list required")
    ids: list[str] = []
    images: list[Any] = []
    labels: list[int] = []
    angles: list[float | None] = []
    for row in rows:
        if set(row) != {"id", "band_1", "band_2", "is_iceberg", "inc_angle"}:
            raise ValueError("Unexpected training schema; no inferred labels")
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("Nonempty image ID required")
        if type(row["is_iceberg"]) is not int or row["is_iceberg"] not in (0, 1):
            raise ValueError("Binary publisher labels required")
        bands = np.asarray([row["band_1"], row["band_2"]], dtype=np.float32)
        if bands.shape != (2, 5625) or not np.isfinite(bands).all():
            raise ValueError("Two finite 75x75 publisher dB bands required")
        angle = row["inc_angle"]
        if angle == "na":
            angles.append(None)
        elif type(angle) in (int, float) and np.isfinite(angle) and 0 < angle < 90:
            angles.append(float(angle))
        else:
            raise ValueError("Invalid incidence angle")
        ids.append(row["id"])
        labels.append(row["is_iceberg"])
        images.append(bands.reshape(2, 75, 75))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs")
    if set(labels) != {0, 1}:
        raise ValueError("Both ship and iceberg classes required")
    return ids, np.stack(images), np.asarray(labels), angles


def audit_groups(
    images: Any, labels: Any, angles: list[float | None]
) -> dict[str, Any]:
    """Union exact/near chips and equal rounded angles before allocating splits.

    Conservative proxies, not recovered acquisition identities. Near search
    includes all eight rotation/reflection transforms in native HH/HV order.
    """
    n = len(images)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    exact: dict[str, int] = {}
    exact_pairs = []
    for i, image in enumerate(images):
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in exact:
            j = exact[digest]
            exact_pairs.append([j, i])
            union(i, j)
        else:
            exact[digest] = i
    # Smooth/subsample only for candidate generation, never for classifier inputs.
    smooth = gaussian_filter(images, sigma=(0, 0, 1.0, 1.0))
    small = smooth.reshape(n, 2, 25, 3, 25, 3).mean(axis=(3, 5))
    flat = small.reshape(n, -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(centered, axis=1, keepdims=True)
    vectors = centered / np.maximum(lengths, 1e-12)
    near: dict[tuple[int, int], dict[str, Any]] = {}
    for turn in range(4):
        for flip in (False, True):
            transformed = np.rot90(small, turn, axes=(2, 3))
            if flip:
                transformed = transformed[:, :, :, ::-1]
            tflat = transformed.reshape(n, -1)
            tv = tflat - tflat.mean(axis=1, keepdims=True)
            tv /= np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-12)
            similarities = vectors @ tv.T
            candidates = np.argwhere(np.triu(similarities >= NEAR_CORRELATION, 1))
            for i, j in candidates:
                original = np.rot90(images[j], turn, axes=(1, 2))
                if flip:
                    original = original[:, :, ::-1]
                rmse = float(np.sqrt(np.mean((images[i] - original) ** 2)))
                key = (int(i), int(j))
                if rmse <= NEAR_RMSE_DB and key not in near:
                    near[key] = {
                        "indices": list(key),
                        "correlation": float(similarities[i, j]),
                        "native_rmse_db": rmse,
                        "rotation_quarters": turn,
                        "reflection": flip,
                        "label_conflict": bool(labels[i] != labels[j]),
                    }
                    union(*key)
    angle_groups: dict[str, list[int]] = {}
    for i, angle in enumerate(angles):
        angle_key = "missing" if angle is None else f"{angle:.4f}"
        angle_groups.setdefault(angle_key, []).append(i)
    for members in angle_groups.values():
        for i in members[1:]:
            union(members[0], i)
    if any(labels[i] != labels[j] for i, j in exact_pairs):
        raise ValueError("Conflicting labels on exact duplicate chips")
    if any(p["label_conflict"] for p in near.values()):
        raise ValueError("Conflicting labels on near-duplicate chips require review")
    groups = [find(i) for i in range(n)]
    repeated = [m for m in angle_groups.values() if len(m) > 1]
    pure_rows = sum(len(m) for m in repeated if len(set(labels[m])) == 1)
    return {
        "groups": groups,
        "exact_duplicate_pairs": exact_pairs,
        "near_duplicate_pairs": list(near.values()),
        "near_definition": {
            "smoothed_correlation_min": NEAR_CORRELATION,
            "native_rmse_db_max": NEAR_RMSE_DB,
            "transforms": "four rotations with/without reflection, no band swaps",
            "limitation": "No guarantee against shifts, resampling or related crops",
        },
        "angle_groups": len(angle_groups),
        "missing_angles": angles.count(None),
        "missing_angle_labels": {
            "ship": sum(a is None and int(y) == 0 for a, y in zip(angles, labels)),
            "iceberg": sum(a is None and int(y) == 1 for a, y in zip(angles, labels)),
        },
        "rows_in_repeated_angle_groups": sum(map(len, repeated)),
        "rows_in_pure_class_repeated_angle_groups": pure_rows,
        "split_proxy_groups": len(set(groups)),
        "largest_proxy_group": max(groups.count(g) for g in set(groups)),
        "acquisition_ids_available": False,
        "coordinates_available": False,
        "acquisition_disjointness_verified": False,
    }


def grouped_split(labels: Any, groups: list[int]) -> list[str]:
    """Prespecified 70/15/15 approximate allocation, no model-result search."""
    roles = ("train", "validation", "holdout")
    fractions = np.asarray([0.70, 0.15, 0.15])
    members = {g: np.flatnonzero(np.asarray(groups) == g) for g in set(groups)}
    if len(members) < 6:
        raise ValueError("Too few independent proxy groups for three partitions")
    rng = np.random.default_rng(SPLIT_SEED)
    order = list(members)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(members[g]))
    totals = np.bincount(labels, minlength=2)
    targets = fractions[:, None] * totals[None, :]
    allocated = np.zeros((3, 2))
    assignment: dict[int, str] = {}
    for g in order:
        counts = np.bincount(labels[members[g]], minlength=2)
        costs = []
        for role in range(3):
            trial = allocated.copy()
            trial[role] += counts
            costs.append(float(np.sum((trial - targets) ** 2 / targets)))
        best = int(np.argmin(costs))
        allocated[best] += counts
        assignment[g] = roles[best]
    split = [assignment[g] for g in groups]
    for split_role in roles:
        if set(labels[np.asarray(split) == split_role]) != {0, 1}:
            raise ValueError("Each partition needs both classes; no silent fallback")
    return split


def train_normalization(images: Any, training: Any) -> tuple[Any, Any]:
    selected = images[training]
    mean = selected.mean(axis=(0, 2, 3), keepdims=True)
    std = selected.std(axis=(0, 2, 3), keepdims=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 1e-6).any():
        raise ValueError("Finite nonconstant training bands required")
    return mean, std


def statistics(images: Any) -> Any:
    flat = images.reshape(len(images), 2, -1)
    return np.concatenate(
        [
            flat.mean(2),
            flat.std(2),
            np.quantile(flat, [0.1, 0.5, 0.9], axis=2)
            .transpose(1, 2, 0)
            .reshape(len(images), -1),
        ],
        axis=1,
    )


def fit_linear(x: Any, y: Any, penalty: float) -> Any:
    design = np.column_stack([np.ones(len(x)), x])

    def objective(w: Any) -> tuple[float, Any]:
        logits = design @ w
        loss = np.mean(np.logaddexp(0, logits) - y * logits)
        loss += penalty * np.sum(w[1:] ** 2) / 2
        gradient = design.T @ (expit(logits) - y) / len(y)
        gradient[1:] += penalty * w[1:]
        return float(loss), gradient

    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Linear baseline failed to converge")
    return result.x


def linear_predict(x: Any, weights: Any) -> Any:
    return expit(np.column_stack([np.ones(len(x)), x]) @ weights)


def log_loss(y: Any, probability: Any) -> float:
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metrics(y: Any, probability: Any) -> dict[str, Any]:
    if (
        len(y) == 0
        or np.shape(y) != np.shape(probability)
        or not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
        or set(y) != {0, 1}
    ):
        raise ValueError("Finite predictions and nonempty labels required")
    predicted = probability >= 0.5
    confusion = [
        [int(np.sum((y == a) & (predicted == b))) for b in (0, 1)] for a in (0, 1)
    ]
    recalls = []
    for index in (0, 1):
        successes, total = confusion[index][index], sum(confusion[index])
        if total == 0:
            raise ValueError("Both classes required for evaluation")
        low = (
            0.0
            if successes == 0
            else float(beta.ppf(0.025, successes, total - successes + 1))
        )
        high = (
            1.0
            if successes == total
            else float(beta.ppf(0.975, successes + 1, total - successes))
        )
        recalls.append({"value": successes / total, "conditional_95_ci": [low, high]})
    return {
        "rows": len(y),
        "log_loss": log_loss(y, probability),
        "accuracy": float(np.mean(predicted == y)),
        "confusion_rows_true_columns_predicted_ship_iceberg": confusion,
        "ship_recall": recalls[0],
        "iceberg_recall": recalls[1],
        "icebergs_called_ships": confusion[1][0],
        "ships_called_icebergs": confusion[0][1],
        "threshold": 0.5,
        "interval_scope": "Conditional chip-binomial; ignores acquisition dependence",
    }


def paired_bootstrap(
    y: Any, baseline: Any, candidate: Any, groups: Any
) -> dict[str, Any]:
    """Paired resampling of complete proxy groups; no scene-independence claim."""
    rng = np.random.default_rng(SPLIT_SEED)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    differences = []
    for _ in range(1000):
        ix = np.concatenate([members[g] for g in rng.choice(unique, len(unique))])
        differences.append(
            [
                log_loss(y[ix], candidate[ix]) - log_loss(y[ix], baseline[ix]),
                float(
                    np.mean((candidate[ix] >= 0.5) == y[ix])
                    - np.mean((baseline[ix] >= 0.5) == y[ix])
                ),
            ]
        )
    bounds = np.quantile(differences, [0.025, 0.975], axis=0)
    return {
        "candidate_minus_linear_log_loss": log_loss(y, candidate)
        - log_loss(y, baseline),
        "log_loss_difference_95_interval": bounds[:, 0].tolist(),
        "candidate_minus_linear_accuracy": float(
            np.mean((candidate >= 0.5) == y) - np.mean((baseline >= 0.5) == y)
        ),
        "accuracy_difference_95_interval": bounds[:, 1].tolist(),
        "replicates": 1000,
        "proxy_groups": len(unique),
        "scope": "Conditional proxy-group bootstrap; actual scene dependence unknown",
    }


def fit_cnn(
    x: Any,
    y: Any,
    train: Any,
    validation: Any,
    *,
    seed: int,
    epochs: int,
    device: str,
) -> tuple[Any, dict[str, Any], Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = nn.Sequential(
        nn.Conv2d(2, 12, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(3),
        nn.Conv2d(12, 24, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(5),
        nn.Flatten(),
        nn.Linear(24 * 5 * 5, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y, dtype=torch.float32, device=device)
    best_loss, best_state, best_epoch = float("inf"), {}, 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        if psutil.Process().memory_info().rss > 2 * 1024**3:
            raise RuntimeError("CPU baseline exceeded 2 GiB process RSS budget")
        if available_commit() < 1024**3:
            raise RuntimeError("System commit headroom exhausted; research stopped")
        model.train()
        order = rng.permutation(train)
        losses = []
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(tx[batch]).flatten(), ty[batch]
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite CNN training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            prediction = model(tx[validation]).flatten().sigmoid().cpu().numpy()
        loss_value = log_loss(y[validation], prediction)
        if not np.isfinite(loss_value):
            raise ValueError("Nonfinite CNN validation loss")
        history.append(
            {
                "epoch": epoch + 1,
                "training_batch_loss_mean": float(np.mean(losses)),
                "validation_log_loss": loss_value,
            }
        )
        if loss_value < best_loss:
            best_loss, best_epoch = loss_value, epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    model.eval()
    # Caller supplies only development chips; holdout inference occurs separately.
    return (
        model,
        {"seed": seed, "selected_epoch": best_epoch, "history": history},
        best_state,
    )


def run(
    source: Path, output: Path, *, epochs: int = 40, device: str = "cpu"
) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    allowed = (REPO_ROOT / "statoil-iceberg-classifier-challenge").resolve()
    if not source.is_relative_to(allowed) or source.name != "train.json":
        raise ValueError("Only project-local competition training archive permitted")
    if output.exists() or not output.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Fresh ignored project data output required")
    if epochs < 1 or epochs > 100 or device != "cpu":
        raise ValueError("Bounded epochs and CPU device required")
    resource_check()
    import torch

    start = time.perf_counter()
    torch.set_num_threads(4)
    ids, images, labels, angles = load_training(source)
    audit = audit_groups(images, labels, angles)
    split = grouped_split(labels, audit["groups"])
    output.mkdir(parents=True)
    provenance = {
        "source_path": str(source.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(source),
        "code_sha256": sha256_file(__file__),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
        "cuda": torch.version.cuda,
        "compute": (
            torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()
        ),
        "licence": (
            "Subject to Kaggle competition rules; "
            "user-provided local archive, acceptance not independently verified"
        ),
        "dataset_url": "https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data",
        "rules_url": "https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/rules",
        "published_region": (
            "Newfoundland and Labrador, per C-CORE authors; "
            "per-chip geography absent"
        ),
    }
    # Freeze partitions before fitting; restricted chip IDs stay local.
    write_json(
        output / "split.json",
        {
            "source_sha256": provenance["source_sha256"],
            "seed": SPLIT_SEED,
            "rows": [
                {"id": i, "group": g, "split": s}
                for i, g, s in zip(ids, audit["groups"], split)
            ],
        },
    )
    write_json(output / "audit.json", audit)
    indices = {
        r: np.flatnonzero(np.asarray(split) == r)
        for r in ("train", "validation", "holdout")
    }
    train, validation, holdout = (
        indices[r] for r in ("train", "validation", "holdout")
    )
    mean, std = train_normalization(images, train)
    normalized = (images - mean) / std
    features = statistics(images)
    feature_mean, feature_std = features[train].mean(0), features[train].std(0)
    features = (features - feature_mean) / np.maximum(feature_std, 1e-6)
    configs = []
    for penalty in (0.001, 0.01, 0.1, 1.0):
        weights = fit_linear(features[train], labels[train], penalty)
        vp = linear_predict(features[validation], weights)
        configs.append((log_loss(labels[validation], vp), penalty, weights))
    _, penalty, weights = min(configs, key=lambda item: item[0])
    np.savez(output / "linear.npz", weights=weights, mean=feature_mean, std=feature_std)
    development = np.concatenate([train, validation])
    remap = {int(original): local for local, original in enumerate(development)}
    local_train = np.asarray([remap[int(i)] for i in train])
    local_validation = np.asarray([remap[int(i)] for i in validation])
    models, histories, validation_predictions = [], [], []
    for seed in SEEDS:
        model, history, state = fit_cnn(
            normalized[development],
            labels[development],
            local_train,
            local_validation,
            seed=seed,
            epochs=epochs,
            device=device,
        )
        torch.save(state, output / f"cnn_{seed}.pth")
        with torch.no_grad():
            vp = (
                model(torch.as_tensor(normalized[validation], device=device))
                .flatten()
                .sigmoid()
                .cpu()
                .numpy()
            )
        models.append(model)
        histories.append(history)
        validation_predictions.append(vp)
    vp_cnn = np.mean(validation_predictions, axis=0)
    vp_linear = linear_predict(features[validation], weights)
    selected = (
        "cnn_ensemble"
        if log_loss(labels[validation], vp_cnn)
        < log_loss(labels[validation], vp_linear)
        else "linear_statistics"
    )
    write_json(
        output / "selection.json",
        {
            "primary": "validation log loss",
            "selected_model": selected,
            "linear_penalty": penalty,
            "cnn_selected_epochs": [h["selected_epoch"] for h in histories],
            "prespecified_classification_threshold": 0.5,
            "epochs_max": epochs,
            "split_sha256": sha256_file(output / "split.json"),
            "band_normalization_mean": mean.flatten().tolist(),
            "band_normalization_std": std.flatten().tolist(),
        },
    )
    # First holdout predictions occur after choices and preprocessing are frozen.
    hp_linear = linear_predict(features[holdout], weights)
    hp_cnn_seeds = []
    with torch.no_grad():
        for model in models:
            hp_cnn_seeds.append(
                model(torch.as_tensor(normalized[holdout], device=device))
                .flatten()
                .sigmoid()
                .cpu()
                .numpy()
            )
    hp_cnn = np.mean(hp_cnn_seeds, axis=0)
    prior = float(labels[train].mean())
    report: dict[str, Any] = {
        "scope": "Exploratory publisher-labelled ship/iceberg chip classification",
        "provenance": provenance,
        "audit": audit,
        "preprocessing": {
            "bands": ["HH", "HV"],
            "input": (
                "publisher float dB; calibration/normalization history "
                "not independently verified"
            ),
            "normalization": "per-band mean/std fitted on training pixels only",
            "resizing": False,
            "augmentation": "none",
            "incidence_angle_model_feature": False,
            "band_quantiles_db": [
                np.quantile(images[:, b], [0, 0.01, 0.5, 0.99, 1]).tolist()
                for b in (0, 1)
            ],
        },
        "splits": {
            r: {
                "rows": len(ix),
                "ships": int(sum(labels[ix] == 0)),
                "icebergs": int(sum(labels[ix] == 1)),
                "groups": len(set(audit["groups"][int(i)] for i in ix)),
            }
            for r, ix in indices.items()
        },
        "validation": {
            "linear_statistics": metrics(labels[validation], vp_linear),
            "cnn_ensemble": metrics(labels[validation], vp_cnn),
        },
        "holdout": {
            "training_prior": metrics(labels[holdout], np.full(len(holdout), prior)),
            "linear_statistics": metrics(labels[holdout], hp_linear),
            "cnn_ensemble": metrics(labels[holdout], hp_cnn),
            "cnn_seeds": [metrics(labels[holdout], p) for p in hp_cnn_seeds],
        },
        "selected_by_validation": selected,
        "training_histories": histories,
        "paired_holdout_comparison": paired_bootstrap(
            labels[holdout], hp_linear, hp_cnn, np.asarray(audit["groups"])[holdout]
        ),
        "linear_grid_validation_log_loss": [
            {"penalty": p, "log_loss": loss} for loss, p, _ in configs
        ],
        "limitations": [
            (
                "No acquisition/time/location metadata: angle/duplicate "
                "groups are proxies, not certified scene/geographic holdouts"
            ),
            "Near search is bounded and can miss shifted/resampled/cropped relatives",
            "Labels are publisher-supplied; no project independent review of chips",
            (
                "Chip classification cannot measure full-scene precision/recall, "
                "false alarms per km2, or AIS identity accuracy"
            ),
            "Transfer to native Sentinel preprocessing is unverified",
            (
                "No operational threshold or promotion; competition test images "
                "and NL locked-test pixels stay unread"
            ),
            (
                "This report consumes the exploratory local holdout; further tuning "
                "cannot reuse it as fresh evidence"
            ),
        ],
        "wall_seconds": time.perf_counter() - start,
        "artifacts": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
    }
    np.savez(
        output / "holdout_predictions.npz",
        labels=labels[holdout],
        indices=holdout,
        groups=np.asarray(audit["groups"])[holdout],
        linear=hp_linear,
        cnn=hp_cnn,
    )
    report["artifacts"]["holdout_predictions.npz"] = sha256_file(
        output / "holdout_predictions.npz"
    )
    write_json(output / "report.json", report)
    return {
        "report": str(output / "report.json"),
        "selected_by_validation": selected,
        "holdout": report["holdout"],
        "splits": report["splits"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(run(args.input, args.output, epochs=args.epochs, device=args.device))
    )


if __name__ == "__main__":
    main()
