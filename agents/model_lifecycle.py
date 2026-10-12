"""Transactional local research model promotion and rollback.

This store is separate from MLflow's non-fatal provenance registry. It never
changes production configuration or enables alerts. Shadow evidence must be
recomputed through the independently reviewed validation interface.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agents.artifacts import REPO_ROOT, sha256_file
from agents.run_versions import digest

DEFAULT_STORE = REPO_ROOT / "state/model_lifecycle.sqlite3"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


class Lifecycle:
    """SQLite transactions keep model pointers and their journal consistent."""

    def __init__(self, path: Path = DEFAULT_STORE):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS packages "
                "(id TEXT PRIMARY KEY, body TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS shadows "
                "(id TEXT PRIMARY KEY, body TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS events "
                "(revision INTEGER PRIMARY KEY, body TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS state "
                "(singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                "revision INTEGER NOT NULL, reference_id TEXT, champion_id TEXT, "
                "threshold REAL)"
            )
            db.execute("INSERT OR IGNORE INTO state VALUES(1, 0, NULL, NULL, NULL)")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def status(self) -> dict[str, Any]:
        with self.transaction() as db:
            row = db.execute("SELECT * FROM state").fetchone()
            assert row is not None
            return {
                **dict(row),
                "scope": "local research only",
                "operational_alerts_enabled": False,
                "events": [
                    json.loads(r["body"])
                    for r in db.execute("SELECT body FROM events ORDER BY revision")
                ],
            }

    def record(self, table: str, identifier: str) -> dict[str, Any]:
        if table not in {"packages", "shadows"}:
            raise ValueError("Unknown record type")
        with self.transaction() as db:
            row = db.execute(
                f"SELECT body FROM {table} WHERE id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown {table} record: {identifier}")
        value: dict[str, Any] = json.loads(row["body"])
        if digest(value) != identifier:
            raise ValueError("Lifecycle record integrity mismatch")
        return value

    def save(self, table: str, value: dict[str, Any]) -> str:
        if table not in {"packages", "shadows"}:
            raise ValueError("Unknown record type")
        identifier = digest(value)
        with self.transaction() as db:
            db.execute(
                f"INSERT OR IGNORE INTO {table} VALUES(?, ?)",
                (identifier, encoded(value)),
            )
        return identifier

    def register(self, weights: Path, metadata: dict[str, Any], *, actor: str) -> str:
        """Register a compatible artifact reference; registration is not approval."""
        if not actor.strip() or any(
            not isinstance(metadata.get(k), str) or not metadata[k].strip()
            for k in ("source", "licence", "experiment", "code_commit")
        ):
            raise ValueError(
                "Actor, source, licence, experiment and code commit required"
            )
        if metadata.get("adapter") != "xview3" or metadata.get("polarizations") != [
            "vv",
            "vh",
        ]:
            raise ValueError("Only the implemented xview3 VV/VH adapter is supported")
        if metadata.get("status") not in {"reference", "exploratory"}:
            raise ValueError("Register only reference or exploratory models")
        if metadata["status"] == "exploratory" and (
            not isinstance(metadata.get("training_dataset_release_sha256"), str)
            or len(metadata["training_dataset_release_sha256"]) != 64
        ):
            raise ValueError("Exploratory models require training release lineage")
        if metadata["status"] == "exploratory" and any(
            sha256_file(p) == metadata["training_dataset_release_sha256"]
            for p in (REPO_ROOT / "datasets/allenai_nl").glob("*/release.json")
        ):
            raise ValueError(
                "Quarantined AllenAI labels cannot supply training lineage"
            )
        checksum = sha256_file(weights)
        if metadata.get("weights_sha256") != checksum:
            raise ValueError("Weights hash mismatch")
        return self.save(
            "packages",
            {
                "schema_version": 1,
                "registered_at": now(),
                "actor": actor,
                "weights": str(weights.resolve()),
                "metadata": metadata,
                "operational_alerts_enabled": False,
            },
        )

    def package(self, identifier: str) -> dict[str, Any]:
        package = self.record("packages", identifier)
        if sha256_file(package["weights"]) != package["metadata"]["weights_sha256"]:
            raise ValueError("Registered weights changed or are unavailable")
        return package

    def _switch(
        self,
        *,
        expected_revision: int,
        action: str,
        target: str,
        threshold: float | None,
        actor: str,
        reason: str,
        evidence: str | None = None,
    ) -> dict[str, Any]:
        if not actor.strip() or not reason.strip():
            raise ValueError("Actor and decision reason required")
        with self.transaction() as db:
            state = db.execute("SELECT * FROM state").fetchone()
            assert state is not None
            if state["revision"] != expected_revision:
                raise ValueError(
                    "Stale lifecycle revision; re-evaluate against current model"
                )
            event = {
                "revision": expected_revision + 1,
                "timestamp": now(),
                "action": action,
                "actor": actor,
                "reason": reason,
                "previous_model": state["champion_id"] or state["reference_id"],
                "previous_threshold": state["threshold"],
                "model": target,
                "threshold": threshold,
                "evidence": evidence,
            }
            if action == "reference":
                if state["reference_id"] is not None:
                    raise ValueError("Reference already initialized")
                db.execute(
                    "UPDATE state SET reference_id=?, revision=?",
                    (target, event["revision"]),
                )
            else:
                db.execute(
                    "UPDATE state SET champion_id=?, threshold=?, revision=?",
                    (target, threshold, event["revision"]),
                )
            db.execute(
                "INSERT INTO events VALUES(?, ?)",
                (event["revision"], encoded(event)),
            )
        return event

    def set_reference(
        self, identifier: str, *, actor: str, reason: str
    ) -> dict[str, Any]:
        if self.package(identifier)["metadata"]["status"] != "reference":
            raise ValueError("Only a declared reference may initialize the baseline")
        return self._switch(
            expected_revision=0,
            action="reference",
            target=identifier,
            threshold=None,
            actor=actor,
            reason=reason,
        )

    def resolve(self) -> dict[str, Any]:
        state = self.status()
        identifier = state["champion_id"] or state["reference_id"]
        if identifier is None:
            raise ValueError("Initialize a reference before shadow evaluation")
        return {
            "model_id": identifier,
            "threshold": state["threshold"],
            "revision": state["revision"],
            "package": self.package(identifier),
            "scope": "local research only",
            "operational_alerts_enabled": False,
        }

    def promote(self, shadow_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        from agents.model_shadow import evaluate_shadow

        shadow = self.record("shadows", shadow_id)
        # Rehash inputs and recompute review/metrics: stored 'passed' is not authority.
        evaluation = evaluate_shadow(self, shadow)
        if evaluation["blockers"]:
            raise ValueError("Promotion blocked: " + "; ".join(evaluation["blockers"]))
        return self._switch(
            expected_revision=shadow["reference_revision"],
            action="promote",
            target=shadow["candidate_id"],
            threshold=evaluation["selected_threshold"],
            actor=actor,
            reason=reason,
            evidence=shadow_id,
        )

    def rollback(
        self, identifier: str, *, expected_revision: int, actor: str, reason: str
    ) -> dict[str, Any]:
        state = self.status()
        if not state["champion_id"] or identifier == state["champion_id"]:
            raise ValueError("Rollback requires a different previously selected model")
        prior = [
            e
            for e in state["events"]
            if e["model"] == identifier and e["action"] in {"reference", "promote"}
        ]
        if not prior:
            raise ValueError("Rollback target was never a reference or promoted model")
        self.package(identifier)
        # A previous package/threshold is restored without retuning on new data.
        return self._switch(
            expected_revision=expected_revision,
            action="rollback",
            target=identifier,
            threshold=prior[-1]["threshold"],
            actor=actor,
            reason=reason,
            evidence=prior[-1]["evidence"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("resolve")
    register = commands.add_parser("register")
    register.add_argument("--weights", type=Path, required=True)
    register.add_argument("--metadata", type=Path, required=True)
    register.add_argument("--actor", required=True)
    for action, flag in (
        ("reference", "model"),
        ("promote", "shadow"),
        ("rollback", "model"),
    ):
        command = commands.add_parser(action)
        command.add_argument(f"--{flag}", required=True)
        command.add_argument("--actor", required=True)
        command.add_argument("--reason", required=True)
        if action == "rollback":
            command.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    lifecycle = Lifecycle(args.store)
    if args.command == "status":
        result = lifecycle.status()
    elif args.command == "resolve":
        result = lifecycle.resolve()
    elif args.command == "register":
        result = {
            "model_id": lifecycle.register(
                args.weights, json.loads(args.metadata.read_text()), actor=args.actor
            )
        }
    elif args.command == "reference":
        result = lifecycle.set_reference(
            args.model, actor=args.actor, reason=args.reason
        )
    elif args.command == "promote":
        result = lifecycle.promote(args.shadow, actor=args.actor, reason=args.reason)
    else:
        result = lifecycle.rollback(
            args.model,
            expected_revision=args.expected_revision,
            actor=args.actor,
            reason=args.reason,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
