"""Command line entry point for JSON-configured KANM simulations.

A configuration either lists ``agents`` and ``edges`` explicitly or generates
them with ``n_agents``/``agent_defaults`` and ``network``/``edge_defaults``.
The README shows the generated form.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .batch import paired_comparison, run_batch
from .model import AgentSpec, EdgeSpec, Simulation
from .networks import agent_ids, agents_from_ids, edges_from_pairs, generate

SCHEMA_VERSION = 1
SIMULATION_KEYS = {
    "truth",
    "steps",
    "trust_points",
    "publish_only_after_inquiry",
    "update_rule",
    "schedule",
    "assertion_rule",
    "trust_model",
    "evidence_policy",
    "count_prior_as_evidence",
}
STRUCTURE_KEYS = {"agents", "edges", "n_agents", "agent_defaults", "network", "edge_defaults"}
ALLOWED_KEYS = SIMULATION_KEYS | STRUCTURE_KEYS | {"seed", "schema_version", "description"}
COMPARABLE_KEYS = SIMULATION_KEYS - {"steps", "trust_points"}


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-finite JSON number {name} is not allowed")


def _load(path: Path) -> dict:
    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle, object_pairs_hook=unique_object, parse_constant=_reject_constant
        )
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a JSON object")
    unknown = sorted(set(payload) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"unknown configuration key(s): {', '.join(unknown)}")
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION or isinstance(version, bool):
        raise ValueError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")
    if ("agents" in payload) == ("n_agents" in payload):
        raise ValueError("configuration must include exactly one of agents or n_agents")
    if "edges" in payload and "network" in payload:
        raise ValueError("use either edges or network, not both")
    if "agents" in payload:
        if not isinstance(payload["agents"], list):
            raise ValueError("agents must be an array")
        if any(not isinstance(item, dict) for item in payload["agents"]):
            raise ValueError("each agents entry must be an object")
        if "agent_defaults" in payload:
            raise ValueError("agent_defaults is only used with n_agents")
    if "edges" in payload:
        if not isinstance(payload["edges"], list):
            raise ValueError("edges must be an array")
        if any(not isinstance(item, dict) for item in payload["edges"]):
            raise ValueError("each edges entry must be an object")
        if "edge_defaults" in payload:
            raise ValueError("edge_defaults is only used with network")
    if "edge_defaults" in payload and "network" not in payload:
        raise ValueError("edge_defaults is only used with network")
    if "agent_defaults" in payload and "n_agents" not in payload:
        raise ValueError("agent_defaults is only used with n_agents")
    for key in ("agent_defaults", "edge_defaults", "network"):
        if key in payload and not isinstance(payload[key], dict):
            raise ValueError(f"{key} must be an object")
    return payload


def build(config: dict) -> tuple[list[AgentSpec], list[EdgeSpec]]:
    """Construct agent and edge specifications from a loaded configuration."""
    if "agents" in config:
        agents = [AgentSpec(**item) for item in config["agents"]]
    else:
        agents = agents_from_ids(agent_ids(config["n_agents"]), **config.get("agent_defaults", {}))
    if "network" in config:
        pairs = generate(config["network"], [a.agent_id for a in agents])
        edges = edges_from_pairs(pairs, **config.get("edge_defaults", {}))
    else:
        edges = [EdgeSpec(**item) for item in config.get("edges", [])]
    return agents, edges


def _parse_compare(text: str) -> tuple[str, list[Any]]:
    key, sep, values = text.partition("=")
    if not sep or not values:
        raise ValueError("--compare expects KEY=VALUE1,VALUE2[,...]")
    if key not in COMPARABLE_KEYS:
        raise ValueError(f"--compare key must be one of {sorted(COMPARABLE_KEYS)}")
    parsed = []
    for raw in values.split(","):
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            parsed.append(raw)
    if len(parsed) < 2 or len({json.dumps(v) for v in parsed}) != len(parsed):
        raise ValueError("--compare needs at least two distinct values")
    return key, parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Laputa-style KANM simulation")
    parser.add_argument("config", type=Path, help="path to simulation JSON config")
    parser.add_argument("--output", type=Path, help="optional result JSON path")
    parser.add_argument("--include-log", action="store_true", help="include the full event log")
    parser.add_argument("--trajectory", action="store_true",
                        help="record mean credence in the truth after every step")
    parser.add_argument("--replicates", type=int, default=1, help="independent seeds to run")
    parser.add_argument(
        "--compare",
        metavar="KEY=V1,V2",
        help="paired comparison on common seeds, e.g. update_rule=laputa,exact_joint",
    )
    args = parser.parse_args(argv)

    try:
        config = _load(args.config)
        agents, edges = build(config)
        if args.replicates < 1:
            raise ValueError("--replicates must be >= 1")
        shared = {key: config[key] for key in SIMULATION_KEYS if key in config}
        seed = config.get("seed", 0)
        if args.compare:
            key, values = _parse_compare(args.compare)
            if args.replicates < 2:
                raise ValueError("--compare needs --replicates >= 2")
            if args.trajectory or args.include_log:
                raise ValueError("--trajectory and --include-log are not supported with --compare")
            shared.pop(key, None)
            conditions = {f"{key}={json.dumps(v)}": {key: v} for v in values}
            result = paired_comparison(
                agents, edges, conditions, replicates=args.replicates, seed_start=seed,
                **shared,
            )
        elif args.replicates == 1:
            result = Simulation(
                agents, edges, seed=seed, record_trajectory=args.trajectory,
                record_events=args.include_log, **shared,
            ).run().as_dict()
            if not args.include_log:
                result.pop("event_log", None)
        else:
            result = run_batch(
                agents, edges, replicates=args.replicates, seed_start=seed,
                record_trajectory=args.trajectory, record_events=args.include_log,
                **shared,
            )
            if not args.include_log:
                for run in result["runs"]:
                    run.pop("event_log", None)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError, UnicodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
