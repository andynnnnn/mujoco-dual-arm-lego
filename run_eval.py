"""Run the remote simulator and expose only observations and action results to the agent."""

import argparse
import base64
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

import paramiko


ROOT = Path(__file__).resolve().parent
MODEL = "gpt-6.2-sol"
EFFORT = "high"


def connect_ssh(client, password=None):
    """Authenticate against a provisioned Linux simulation host."""
    client.load_system_host_keys()
    known_hosts = os.environ.get("ROBOT_SIM_KNOWN_HOSTS")
    if known_hosts:
        client.load_host_keys(known_hosts)
    client.connect(os.environ["ROBOT_SIM_HOST"],
                   port=int(os.environ.get("ROBOT_SIM_PORT", "22")),
                   username=os.environ.get("ROBOT_SIM_USER"),
                   password=password, timeout=10)
    client.get_transport().set_keepalive(15)


def model_action(directory, observation, detail, history, remaining_actions, timeout, pose_study=False):
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("未找到 Codex CLI")
    prompt = (
        "Count Legos with two Panda arms. End with exactly two red and three orange bricks in the tilted dark "
        "bin and nothing else. Choose exactly ONE next action from the latest images. The 768x768 overhead is "
        "the first image; an optional second image is the wrist close-up. Coordinates are on the first image, "
        "u right, v down, interpolating the numbered grid. For a brick, target its visible top center. "
        "angle is degrees from the downward image vertical toward the right, modulo 180. "
        "For example a long axis that goes down-right is +45 and down-left is -45. "
        "Use left for u<384, right for u>=384 when reachable. Use both when feasible. "
        "kind=pick puts a visible red or orange in the bin; specify its exact color. Never exceed the quotas. "
        "Some red bricks may be underneath distractors. If needed, kind=separate moves one clearly visible "
        "blue or green blocker to an empty tabletop point you choose from the current overhead. "
        "For separate, give drop_u and drop_v on that same image. Choose open tabletop on the selected arm's "
        "side, outside the brick cluster, roughly 110 pixels or more from every other brick and at least "
        "85 pixels from the bin walls. Leave room for the gripper and stay away from the image and table edges. "
        "Keep the brick's grasp orientation for the move. For all other actions set drop_u and drop_v to null. "
        "Prefer accessible targets first. For unclear color, layering, or angle, kind=inspect positions the "
        "wrist camera over that pixel without grasping. Don't target an exposed edge of a covered brick; move "
        "the visible upper brick or inspect. Any bump can move pieces, so use the current coordinates "
        "instead of any previous target. A recent rejection calls for a materially different action. "
        "If the force feedback stopped on a neighboring brick, clear the obstructing distractor, try a different "
        "arm or choose a different exposed target. An arm reported holding a brick is unavailable for new picks. "
        "kind=done only if the quota is reached or no safe progress remains. Set unused fields to placeholders. "
        f"{remaining_actions} action slots remain. Return the structured action only.\n"
        f"Action history: {json.dumps(history[-12:])}"
    )
    schema = "action_schema.json"
    if pose_study:
        schema = "pose_action_schema.json"
        prompt += (
            "\nFor pick and separate provide integer grasp_height_mm (3 through 65): the height of the "
            "center of the gripped brick body above the tabletop, not the top face or the stud tips. "
            "The same brick type is about 22 mm tall before studs; an upright brick on the table is "
            "usually grasped around 12 mm, and an upright brick on top of one other brick around 34 mm. "
            "Estimate this from the image and use inspect when the layer is unclear. For inspect or done "
            "set grasp_height_mm to null."
        )
    # Use the existing Codex ChatGPT sign-in rather than forwarding a configured third-party API endpoint.
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in {"OPENAI_API_KEY", "OPENAI_BASE_URL"}}
    images = [directory / observation]
    if detail:
        images.append(directory / detail)
    with (tempfile.TemporaryDirectory(prefix="lego_decision_") if pose_study else nullcontext(directory)) as workdir:
        if pose_study:
            images = [shutil.copyfile(image, Path(workdir) / image.name) for image in images]
        result = subprocess.run(
            [codex, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
             "--sandbox", "read-only", "--disable", "shell_tool", "--disable", "multi_agent",
             "-m", MODEL, "-c", f'model_reasoning_effort="{EFFORT}"', "--color", "never",
             "--output-schema", str(ROOT / schema), "-C", str(workdir),
             "--image", *(str(image) for image in images)],
            input=prompt, text=True, encoding="utf-8", errors="replace", capture_output=True,
            timeout=max(1, min(timeout, 120)), env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if result.returncode != 0:
        raise RuntimeError(f"GPT-6.2 Sol High 调用失败（Codex 退出码 {result.returncode}）")
    if f"model: {MODEL}" not in result.stderr or f"reasoning effort: {EFFORT}" not in result.stderr:
        raise RuntimeError("Codex 未确认指定的模型和推理档位")
    action = json.loads(result.stdout)
    if action["kind"] == "done":
        return {"done": True}
    separate = action["kind"] == "separate"
    if (action["arm"] not in ("left", "right") or not -180 <= action["angle"] <= 180
            or not all(type(action[axis]) is int and 0 <= action[axis] < 768 for axis in ("u", "v"))
            or (action["kind"] == "pick" and action["color"] not in ("red", "orange"))
            or (separate and action["color"] not in ("blue", "green"))):
        raise ValueError("模型返回的取放目标无效")
    if separate and not all(type(action[axis]) is int and 0 <= action[axis] < 768 for axis in ("drop_u", "drop_v")):
        raise ValueError("模型返回的临时落点无效")
    if pose_study and action["kind"] in ("pick", "separate"):
        height = action["grasp_height_mm"]
        if type(height) is not int or not 3 <= height <= 65:
            raise ValueError("模型返回的抓取高度无效")
    return action


def write_json(path, body):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(body, indent=2), encoding="utf-8")
    publish_file(temp, path)


def publish_file(source, destination):
    for attempt in range(5):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.015)


def episode(client, directory, seed, run, number, action_limit, seconds, live, auto=False,
            pose_policy=None, initial_action=None):
    remote = os.environ["ROBOT_SIM_REMOTE_DIR"].rstrip("/")
    remote_output = f"{remote}/runs/{run}/trial_{number:02d}"
    python = os.environ.get("ROBOT_SIM_PYTHON", "python3")
    arguments = [
        "env", "MUJOCO_GL=egl", python, remote + "/experiment.py",
        "--seed", str(seed), "--output", remote_output,
    ]
    if live:
        arguments.append("--live")
    if auto:
        arguments.extend(("--clutter", "--force-feedback"))
    if pose_policy:
        arguments.extend(("--pose-policy", pose_policy))
    stdin, stdout, stderr = client.exec_command(shlex.join(arguments), timeout=seconds)
    status = {"run": run, "trial": number, "seed": seed, "phase": "starting", "score": 0,
              "max_score": 10, "counts_in_bin": {}, "action": 0}
    if auto:
        status.update(model=MODEL, reasoning_effort=EFFORT, scene="clutter", force_feedback=True)
    if pose_policy:
        status["pose_policy"] = pose_policy

    def update(**fields):
        status.update(fields)
        if live:
            write_json(live / "state.json", status)

    update()
    directory.mkdir()
    transcript = []
    history = []
    model_waits = []

    def send_action(action):
        if live and pose_policy == "predicted":
            action = {**action, "recording": {"model_calls": len(model_waits),
                                              "model_seconds": round(sum(model_waits), 2)}}
        stdin.write(json.dumps(action) + "\n")
        stdin.flush()

    deadline = time.monotonic() + seconds
    started = time.monotonic()
    termination = "action_budget"
    last_telemetry = 0.
    with client.open_sftp() as sftp:
        for index in range(action_limit + 1):
            while True:
                response = stdout.readline()
                if not response:
                    raise RuntimeError(stderr.read().decode("utf-8", "replace"))
                reply = json.loads(response)
                if "frame" in reply:
                    frame = live / "frame.tmp"
                    frame.write_bytes(base64.b64decode(reply["frame"]))
                    try:
                        frame.replace(live / "frame.jpg")
                    except PermissionError:
                        pass  # Retain the previous frame if a browser was reading it.
                    reading = reply.get("force")
                    now = time.monotonic()
                    if reading and now - last_telemetry >= .1:
                        update(force_reading=reading)
                        last_telemetry = now
                elif "phase" in reply:
                    update(phase=reply["phase"], arm=reply["arm"], action=index,
                           force_reading=reply.get("force"))
                else:
                    break
            name = reply["observation"]
            sftp.get(remote_output + "/" + name, str(directory / name))
            detail = reply.get("detail")
            if detail:
                sftp.get(remote_output + "/" + detail, str(directory / detail))
            progress = reply.pop("progress", {})
            if auto and index:
                history.append({"action": transcript[-1], **{k: v for k, v in reply.items() if k not in ("observation", "detail")}})
            if live:
                top = live / "top.tmp"
                top.write_bytes((directory / name).read_bytes())
                publish_file(top, live / "top.jpg")
                if detail:
                    wrist = live / "detail.tmp"
                    wrist.write_bytes((directory / detail).read_bytes())
                    publish_file(wrist, live / "detail.jpg")
                force = reply.get("force") or {}
                events = force.get("events", [])
                update(**progress, phase="planning" if auto else "deciding", action=index, arm=None,
                       placed=reply.get("placed"), observation=index, detail=bool(detail),
                       separated=reply.get("separated", False), drop_pixel=reply.get("drop_pixel"),
                       force_summary=force, last_force_event=events[-1] if events else status.get("last_force_event"),
                       shifted_neighbors=reply.get("shifted_neighbors", 0), last_error=reply.get("error"))
            write_json(directory / f"step_{index:02d}.json", reply)
            if pose_policy and index:
                print(f"STEP {pose_policy} seed={seed} {index}/{action_limit} "
                      f"{transcript[-1]['kind']} score={progress.get('score', 0)}", flush=True)
            if index == action_limit:
                break
            action_file = directory / f"action_{index + 1:02d}.json"
            while not auto and not action_file.exists() and time.monotonic() < deadline:
                time.sleep(.2)
            if time.monotonic() >= deadline:
                termination = "time_limit"
                break
            if auto:
                try:
                    if progress.get("success"):
                        action = {"done": True}
                    else:
                        if reply.get("shifted_neighbors") or reply.get("error"):
                            update(phase="replanning")
                        planning_started = time.monotonic()
                        if initial_action is not None and index == 0:
                            action = initial_action(directory, name, detail, history, action_limit,
                                                    deadline - planning_started)
                        else:
                            action = model_action(directory, name, detail, history, action_limit - index,
                                                  deadline - planning_started, pose_study=bool(pose_policy))
                        model_waits.append(round(time.monotonic() - planning_started, 2))
                        write_json(directory / f"plan_{len(model_waits):02d}.json",
                                   {"observation": name, "detail": detail,
                                    "model_seconds": model_waits[-1], "action": action})
                        if not action.get("done"):
                            action["observation"] = index
                        if live and pose_policy and action.get("kind") in ("pick", "separate"):
                            status["pose_target"] = action.copy()
                        update(model_calls=len(model_waits), model_seconds=round(sum(model_waits), 2))
                except Exception as exc:
                    update(phase="error", error=str(exc))
                    send_action({"done": True})
                    stdout.channel.recv_exit_status()
                    raise
                write_json(action_file, action)
            else:
                action = json.loads(action_file.read_text(encoding="utf-8-sig"))
            transcript.append(action)
            if action.get("done") is True:
                termination = "goal_reached" if pose_policy and progress.get("success") else "agent_done"
                break
            send_action(action)
        send_action({"done": True})
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(stderr.read().decode("utf-8", "replace"))
        result = ROOT / "results" / run / directory.name
        result.mkdir(parents=True, exist_ok=True)
        for name in ("episode.mp4", "final_overview.png", "final_overhead.png"):
            sftp.get(remote_output + "/" + name, str(result / name))
        with sftp.open(remote_output + "/score.json", "rb") as score_file:
            score = json.load(score_file)
        score.update(seed=seed, termination=termination, agent_actions=transcript,
                     elapsed_seconds=round(time.monotonic() - started, 2))
        if auto:
            score.update(model=MODEL, reasoning_effort=EFFORT, model_calls=len(model_waits),
                         model_seconds=round(sum(model_waits), 2), model_waits=model_waits)
        write_json(result / "score.json", score)
        update(phase="finished", success=score["success"], score=score["score"],
               counts_in_bin=score["counts_in_bin"], checkpoints=score["checkpoints"],
               termination=termination, action=sum(not action.get("done") for action in transcript))
        write_json(directory / "finished.json", {"termination": termination})
        print(directory.name, score["score"], score["success"], flush=True)
        return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("seeds", type=int, nargs="+")
    parser.add_argument("--actions", type=int, default=12)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.run.isascii() or not args.run.replace("_", "").isalnum():
        parser.error("Use letters, digits and underscores for the run name")
    observations = ROOT / "observations" / args.run
    observations.mkdir(parents=True)
    live = ROOT / "live" if args.live else None
    if live:
        live.mkdir(exist_ok=True)
    with paramiko.SSHClient() as ssh:
        connect_ssh(ssh, os.environ.pop("ROBOT_SIM_SSH_PASSWORD", None))
        try:
            scores = [episode(ssh, observations / f"trial_{i:02d}", seed, args.run, i, args.actions, args.seconds, live)
                      for i, seed in enumerate(args.seeds, 1)]
        except Exception:
            if live:
                write_json(live / "state.json", {"phase": "error", "run": args.run})
            raise
    write_json(ROOT / "results" / args.run / "summary.json",
               {"trials": len(scores), "successes": sum(s["success"] for s in scores),
                "mean_score": sum(s["score"] for s in scores) / len(scores),
                "scores": [s["score"] for s in scores]})


if __name__ == "__main__":
    main()
