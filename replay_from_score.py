"""Replay a recorded dual-arm B episode without invoking the model."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.score.read_bytes())
    if record.get("pose_policy") != "predicted":
        parser.error("expected a B-group score record")
    actions = [item for item in record["agent_actions"] if not item.get("done")]
    waits = record["model_waits"]
    commands = [
        {**action, "recording": {"model_calls": min(index, len(waits)),
                                 "model_seconds": round(sum(waits[:index]), 2)}}
        for index, action in enumerate(actions, 1)
    ]
    commands.append({"done": True, "recording": {"model_calls": record["model_calls"],
                                                 "model_seconds": record["model_seconds"]}})
    script = Path(__file__).with_name("experiment.py")
    subprocess.run([sys.executable, str(script), "--seed", str(record["seed"]), "--output", str(args.output),
                    "--clutter", "--force-feedback", "--pose-policy", "predicted", "--record-dashboard"],
                   input="".join(json.dumps(command) + "\n" for command in commands), text=True, check=True)


if __name__ == "__main__":
    main()
