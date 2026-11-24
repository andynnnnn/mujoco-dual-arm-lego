"""Loopback dashboard for the B-group dual-arm simulation."""
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import threading

import paramiko
import run_eval

LEGO_LIVE = run_eval.ROOT / "live"
LEGO_POSE_POLICY = "predicted"


def run_lego_trial(server, run, seed):
    try:
        observations = run_eval.ROOT / "observations" / run / "trial_01"
        observations.parent.mkdir(parents=True)
        score = run_eval.episode(server.ssh, observations, seed, run, 1, 12, 900, LEGO_LIVE, auto=True,
                                 pose_policy=LEGO_POSE_POLICY)
        run_eval.write_json(run_eval.ROOT / "results" / run / "summary.json",
                            {"trials": 1, "successes": int(score["success"]), "mean_score": score["score"],
                             "model": score["model"], "reasoning_effort": score["reasoning_effort"],
                             "pose_policy": score["pose_policy"]})
    except Exception as exc:
        print(f"Count Legos run {run} failed: {exc}", file=sys.stderr, flush=True)
        status = json.loads((LEGO_LIVE / "state.json").read_bytes())
        status.update(phase="error", error=status.get("error") or "仿真或模型调用中断，请重试。")
        run_eval.write_json(LEGO_LIVE / "state.json", status)
    finally:
        server.lego_lock.release()


class Viewer(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.headers.get("Host") != "127.0.0.1:53856":
            self.send_error(403)
            return
        if self.path in ("/", "/legos"):
            page = Path(__file__).with_name("legos.html").read_bytes()
            self.send_bytes(page.replace(b"__TOKEN__", self.server.token.encode()), "text/html; charset=utf-8")
            return
        if self.path == "/legos/state":
            status_file = LEGO_LIVE / "state.json"
            if status_file.is_file():
                self.send_asset(status_file, "application/json")
            else:
                self.send_json(200, {"phase": "idle"})
            return
        if self.path.split("?", 1)[0] == "/legos/top.jpg":
            self.send_asset(LEGO_LIVE / "top.jpg", "image/jpeg")
            return
        if self.path.split("?", 1)[0] == "/legos/detail.jpg":
            self.send_asset(LEGO_LIVE / "detail.jpg", "image/jpeg")
            return
        if self.path == "/legos/frame.jpg":
            self.send_asset(LEGO_LIVE / "frame.jpg", "image/jpeg")
            return
        if self.path == "/legos/replay.mp4":
            state = json.loads((LEGO_LIVE / "state.json").read_bytes()) if (LEGO_LIVE / "state.json").is_file() else {}
            run, trial = state.get("run", ""), state.get("trial")
            if state.get("phase") != "finished" or not run.replace("_", "").isalnum() or not isinstance(trial, int):
                self.send_error(404)
                return
            result = LEGO_LIVE.parent / "results" / run / f"trial_{trial:02d}"
            annotated = result / "episode_annotated.mp4"
            self.send_asset(annotated if annotated.is_file() else result / "episode.mp4", "video/mp4")
            return
        self.send_error(404)

    def send_asset(self, path, content_type):
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        except PermissionError:
            self.send_error(503)
            return
        self.send_bytes(payload, content_type)

    def send_bytes(self, payload, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, status, value):
        payload = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if (self.path != "/legos/start" or self.headers.get("Host") != "127.0.0.1:53856"
                or self.headers.get("Origin") != "http://127.0.0.1:53856"
                or not secrets.compare_digest(self.headers.get("X-Panda-Token", ""), self.server.token)):
            self.send_error(403)
            return
        self.start_legos()

    def start_legos(self):
        if not self.server.lego_lock.acquire(blocking=False):
            self.send_json(409, {"error": "本轮进行中，请等实验结束。"})
            return
        run = f"sol_high_b_{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"
        seed = secrets.randbits(32)
        try:
            LEGO_LIVE.mkdir(parents=True, exist_ok=True)
            run_eval.write_json(LEGO_LIVE / "state.json", {"run": run, "trial": 1, "phase": "starting",
                                "seed": seed, "score": 0, "counts_in_bin": {}, "action": 0,
                                "pose_policy": LEGO_POSE_POLICY})
            threading.Thread(target=run_lego_trial, args=(self.server, run, seed), daemon=True).start()
        except Exception:
            self.server.lego_lock.release()
            self.send_json(500, {"error": "启动仿真实验失败。"})
            return
        self.send_json(202, {"run": run, "seed": seed})


def main():
    ssh = paramiko.SSHClient()
    run_eval.connect_ssh(ssh, os.environ.pop("ROBOT_SIM_SSH_PASSWORD", None))
    server = ThreadingHTTPServer(("127.0.0.1", 53856), Viewer)
    server.daemon_threads = True
    server.ssh = ssh
    server.token = secrets.token_urlsafe(32)
    server.lego_lock = threading.Lock()
    status_file = LEGO_LIVE / "state.json"
    if status_file.is_file():
        status = json.loads(status_file.read_bytes())
        if status.get("phase") not in ("finished", "error", "idle"):
            status.update(phase="error", error="上一轮在服务重启时结束，可以再跑一轮。")
            run_eval.write_json(status_file, status)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        ssh.close()


if __name__ == "__main__":
    main()
