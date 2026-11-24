"""Run paired Sol High episodes with and without simulated object pose correction."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from statistics import median
from threading import Lock
import time

import paramiko

import run_eval


POLICIES = ("corrected", "predicted")


def share_first_decision(path):
    lock = Lock()

    def decide(directory, observation, detail, history, remaining, timeout):
        fingerprint = hashlib.sha256((directory / observation).read_bytes()).hexdigest()
        with lock:
            if path.is_file():
                cached = json.loads(path.read_bytes())
                if fingerprint != cached["image_sha256"]:
                    raise RuntimeError("Paired episodes have different initial camera frames")
                return cached["action"]
            started = time.monotonic()
            action = run_eval.model_action(directory, observation, detail, history, remaining, timeout, pose_study=True)
            run_eval.write_json(path, {"image_sha256": fingerprint, "action": action,
                                       "model_seconds": round(time.monotonic() - started, 2)})
            return action.copy()

    return decide


def run_trial(run, policy, number, seed, password, actions, seconds, first_decision):
    group = f"{run}/{policy}"
    trial = f"trial_{number:02d}"
    score_path = run_eval.ROOT / "results" / group / trial / "score.json"
    if score_path.is_file():
        return json.loads(score_path.read_bytes())
    observations = run_eval.ROOT / "observations" / group / trial
    observations.parent.mkdir(parents=True, exist_ok=True)
    if observations.exists():
        observations.rename(observations.with_name(f"{trial}_interrupted_{time.time_ns()}"))
    print(f"START {policy} seed={seed}", flush=True)
    with paramiko.SSHClient() as ssh:
        run_eval.connect_ssh(ssh, password)
        score = run_eval.episode(ssh, observations, seed, group, number, actions, seconds, None, auto=True,
                                 pose_policy=policy, initial_action=first_decision)
    print(f"DONE {policy} seed={seed} score={score['score']}/10", flush=True)
    return score


def predicted_errors(record):
    prediction = record["prediction_xyz_m"]
    reference = record["reference_xyz_m"]
    return {
        "xy_mm": math.dist(prediction[:2], reference[:2]) * 1000,
        "z_mm": abs(prediction[2] - reference[2]) * 1000,
        "yaw_deg": abs((record["prediction_yaw_deg"] - record["reference_yaw_deg"] + 90) % 180 - 90),
    }


def metrics(score, observations):
    picks = [attempt for attempt in score["pose_records"] if attempt["kind"] == "pick"]
    errors = [predicted_errors(attempt) for attempt in picks if "reference_xyz_m" in attempt]
    findings = [json.loads(path.read_bytes()) for path in sorted(observations.glob("step_*.json"))]
    stops = [event for step in findings for event in (step.get("force") or {}).get("events", [])]
    return {"seed": score["seed"], "score": score["score"], "success": score["success"],
            "first_pick_lifted": bool(picks and picks[0].get("lifted")),
            "pick_attempts": len(picks), "target_lifts": sum(bool(pick.get("lifted")) for pick in picks),
            "force_stops": len(stops), "force_events": stops, "termination": score["termination"],
            "elapsed_seconds": score["elapsed_seconds"], "model_calls": score["model_calls"],
            "predicted_pose_errors": errors}


def report(root, rows, seeds):
    groups = {}
    for policy in POLICIES:
        group = [rows[seed, policy] for seed in seeds]
        errors = [error for row in group for error in row["predicted_pose_errors"]]
        groups[policy] = {
            "trials": group, "mean_score": sum(row["score"] for row in group) / len(group),
            "successes": sum(row["success"] for row in group),
            "first_pick_successes": sum(row["first_pick_lifted"] for row in group),
            "target_lifts": sum(row["target_lifts"] for row in group),
            "pick_attempts": sum(row["pick_attempts"] for row in group),
            "force_stops": sum(row["force_stops"] for row in group),
            "median_prediction_error": {axis: round(median(e[axis] for e in errors), 2) if errors else None
                                        for axis in ("xy_mm", "z_mm", "yaw_deg")},
        }
    run_eval.write_json(root / "comparison.json", {
        "model": run_eval.MODEL, "reasoning_effort": run_eval.EFFORT, "seeds": seeds,
        "force_feedback": True, "shared_first_decision": True, "groups": groups,
        "paired_score_delta_b_minus_a": {str(seed): rows[seed, "predicted"]["score"] - rows[seed, "corrected"]["score"]
                                         for seed in seeds},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actions", type=int, default=12)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    if not args.run.isascii() or not args.run.replace("_", "").isalnum():
        parser.error("Use letters, digits and underscores for the run name")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct seeds")
    password = os.environ.pop("ROBOT_SIM_SSH_PASSWORD", None)
    root = run_eval.ROOT / "results" / args.run
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": args.seeds, "actions": args.actions, "seconds": args.seconds,
                "model": run_eval.MODEL, "reasoning_effort": run_eval.EFFORT}
    manifest_path = root / "protocol.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_bytes()) != manifest:
        parser.error("Use a new run name when changing the protocol")
    run_eval.write_json(manifest_path, manifest)
    rows, failures = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for number, seed in enumerate(args.seeds, 1):
            first = share_first_decision(root / f"initial_{seed}.json")
            for policy in POLICIES if number % 2 else reversed(POLICIES):
                future = executor.submit(run_trial, args.run, policy, number, seed, password, args.actions, args.seconds, first)
                futures[future] = (seed, policy, number)
        for future in as_completed(futures):
            seed, policy, number = futures[future]
            try:
                score = future.result()
                observations = run_eval.ROOT / "observations" / args.run / policy / f"trial_{number:02d}"
                rows[seed, policy] = metrics(score, observations)
            except Exception as error:
                failures[f"{seed}/{policy}"] = str(error)
                print(f"ERROR {policy} seed={seed}: {error}", flush=True)
    if failures:
        run_eval.write_json(root / "interrupted.json", failures)
        raise SystemExit(1)
    report(root, rows, args.seeds)
    print(root / "comparison.json", flush=True)


if __name__ == "__main__":
    main()
