"""
bridge_ai_server.py
-------------------
Local HTTP bridge for city_visual.html to control external Python AI.

Endpoints
---------
GET  /health
GET  /status
POST /train
POST /stop
POST /apply

Usage
-----
python bridge_ai_server.py --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from environment import TrafficEnv
from grid import EMPTY as PY_EMPTY
from grid import HIGHWAY as PY_HIGHWAY
from grid import INTERSECTION as PY_INTER
from grid import ROAD as PY_ROAD
from rl_trainer import LinearDoubleQAgent

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "linear_double_q_weights.npz"
DEFAULT_LOG_PATH = ROOT / "models" / "bridge_train.log"


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Terminate training process and any child workers."""
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    proc.terminate()


@dataclass
class BridgeState:
    process: Optional[subprocess.Popen] = None
    started_at: float = 0.0
    cmd: list[str] = field(default_factory=list)
    log_path: Path = DEFAULT_LOG_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    last_exit_code: Optional[int] = None
    last_error: str = ""


STATE = BridgeState()
LOCK = threading.Lock()


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(data)


def _read_log_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _watch_process(proc: subprocess.Popen) -> None:
    code = proc.wait()
    with LOCK:
        if STATE.process is proc:
            STATE.last_exit_code = code
            STATE.process = None


def _is_running() -> bool:
    return STATE.process is not None and STATE.process.poll() is None


def _current_status() -> Dict[str, Any]:
    with LOCK:
        running = _is_running()
        return {
            "running": running,
            "pid": STATE.process.pid if running and STATE.process else None,
            "started_at": STATE.started_at,
            "cmd": STATE.cmd,
            "log_path": str(STATE.log_path),
            "model_path": str(STATE.model_path),
            "last_exit_code": STATE.last_exit_code,
            "last_error": STATE.last_error,
            "log_tail": _read_log_tail(STATE.log_path),
            "model_exists": STATE.model_path.exists(),
        }


def _to_python_grid(html_grid: list[list[int]]) -> np.ndarray:
    h = len(html_grid)
    w = len(html_grid[0]) if h else 0
    arr = np.zeros((h, w), dtype=np.int8)

    for y in range(h):
        for x in range(w):
            v = int(html_grid[y][x])
            if v == 3:
                arr[y, x] = PY_HIGHWAY
            elif v == 4:
                arr[y, x] = PY_INTER
            elif v in (1, 2):
                arr[y, x] = PY_ROAD
            else:
                arr[y, x] = PY_EMPTY

    return arr


def _infer_dir_grid(html_grid: list[list[int]]) -> list[list[Optional[str]]]:
    h = len(html_grid)
    w = len(html_grid[0]) if h else 0
    out: list[list[Optional[str]]] = [[None for _ in range(w)] for _ in range(h)]

    def drivable(v: int) -> bool:
        return v in (1, 2, 3, 4)

    for y in range(h):
        for x in range(w):
            v = int(html_grid[y][x])
            if v == 0:
                out[y][x] = None
                continue
            if v == 4:
                out[y][x] = "x"
                continue

            h_count = 0
            v_count = 0
            if x > 0 and drivable(int(html_grid[y][x - 1])):
                h_count += 1
            if x + 1 < w and drivable(int(html_grid[y][x + 1])):
                h_count += 1
            if y > 0 and drivable(int(html_grid[y - 1][x])):
                v_count += 1
            if y + 1 < h and drivable(int(html_grid[y + 1][x])):
                v_count += 1
            out[y][x] = "h" if h_count >= v_count else "v"

    return out


def _build_inter_mask(html_grid: list[list[int]]) -> list[list[int]]:
    h = len(html_grid)
    w = len(html_grid[0]) if h else 0
    out = [[0 for _ in range(w)] for _ in range(h)]

    def cell_bit(v: int) -> int:
        if v == 3:
            return 4
        if v in (1, 2):
            return 1
        if v == 4:
            return 1
        return 0

    for y in range(h):
        for x in range(w):
            v = int(html_grid[y][x])
            if v != 4:
                out[y][x] = 0
                continue

            bits = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    bits |= cell_bit(int(html_grid[ny][nx]))
            out[y][x] = bits if bits else 1

    return out


def _to_html_grid(py_grid: np.ndarray) -> list[list[int]]:
    h, w = py_grid.shape
    out = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            v = int(py_grid[y, x])
            if v == PY_HIGHWAY:
                out[y][x] = 3
            elif v == PY_INTER:
                out[y][x] = 4
            elif v == PY_ROAD:
                out[y][x] = 1
            else:
                out[y][x] = 0

    return out


def _sampled_best_action(
    agent: LinearDoubleQAgent,
    obs: np.ndarray,
    info: Dict[str, Any],
    step: int,
    candidate_actions: int,
) -> int:
    total_actions = int(agent.env.action_space_n)
    candidate_actions = max(32, min(candidate_actions, total_actions))

    if candidate_actions >= total_actions:
        return agent.act(obs, info, step, epsilon=0.0)

    candidates = agent.rng.choice(total_actions, size=candidate_actions, replace=False)
    best_a = int(candidates[0])
    best_v = -float("inf")

    for a in candidates:
        aa = int(a)
        v = agent.q_sum(obs, info, step, aa)
        if v > best_v:
            best_v = v
            best_a = aa

    return best_a


def _apply_trained_policy(
    state: Dict[str, Any],
    model_path: Path,
    steps: int,
    candidate_actions: int,
) -> Dict[str, Any]:
    width = int(state.get("width", 0))
    height = int(state.get("height", 0))
    html_grid = state.get("grid")

    if not isinstance(html_grid, list) or not html_grid:
        raise ValueError("Invalid state.grid")

    env = TrafficEnv(
        width=width,
        height=height,
        max_cars=120,
        spawn_rate=0.35,
        episode_length=max(1, steps),
    )

    py_grid = _to_python_grid(html_grid)
    env.grid.grid = py_grid.copy()
    env.engine.reset()
    env._t = 0
    env._last_metrics = env.engine.step()
    env._prev_metrics = dict(env._last_metrics)

    agent = LinearDoubleQAgent(env, alpha=0.0, gamma=0.98, seed=42)
    agent.load(str(model_path))

    obs = env._build_observation()
    info = dict(env._last_metrics)

    total_reward = 0.0
    for step_idx in range(max(1, steps)):
        action = _sampled_best_action(agent, obs, info, step_idx, candidate_actions)
        next_obs, reward, done, next_info = env.step(action)
        total_reward += float(reward)
        obs = next_obs
        info = next_info
        if done:
            break

    out_state = dict(state)
    out_grid = _to_html_grid(env.grid.grid)
    out_state["grid"] = out_grid
    out_state["dirGrid"] = _infer_dir_grid(out_grid)
    out_state["interMask"] = _build_inter_mask(out_grid)

    # Keep existing zones/buildings/city mask if present; ensure fallback values.
    if "cityMask" not in out_state or not out_state["cityMask"]:
        out_state["cityMask"] = [[True for _ in range(width)] for _ in range(height)]
    if "zone" not in out_state or not out_state["zone"]:
        out_state["zone"] = [[0 for _ in range(width)] for _ in range(height)]
    if "buildingCache" not in out_state or not isinstance(out_state["buildingCache"], list):
        out_state["buildingCache"] = []

    return {
        "state": out_state,
        "stats": {
            "steps": max(1, steps),
            "candidate_actions": candidate_actions,
            "total_reward": total_reward,
        },
    }


class BridgeHandler(BaseHTTPRequestHandler):
    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        _json_response(self, HTTPStatus.OK, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, HTTPStatus.OK, {"ok": True, "service": "bridge_ai_server"})
            return

        if self.path == "/status":
            _json_response(self, HTTPStatus.OK, {"ok": True, **_current_status()})
            return

        _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/train":
                self._handle_train()
                return
            if self.path == "/stop":
                self._handle_stop()
                return
            if self.path == "/apply":
                self._handle_apply()
                return

            _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
        except Exception as exc:  # noqa: BLE001
            with LOCK:
                STATE.last_error = str(exc)
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    def _handle_train(self) -> None:
        body = self._read_json()

        train_mode = str(body.get("mode", "single")).strip().lower()
        if train_mode not in {"single", "parallel"}:
            _json_response(
                self,
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Invalid mode. Use 'single' or 'parallel'."},
            )
            return

        episodes = int(body.get("episodes", 1000))
        episode_length = int(body.get("episode_length", 250))
        eval_every = int(body.get("eval_every", 25))
        eval_episodes = int(body.get("eval_episodes", 3))
        seed = int(body.get("seed", 42))
        runs = int(body.get("runs", 8))
        max_parallel = int(body.get("max_parallel", runs))
        numpy_threads = int(body.get("numpy_threads", 1))
        save_path_raw = str(body.get("save_path", str(DEFAULT_MODEL_PATH)))

        save_path = Path(save_path_raw)
        if not save_path.is_absolute():
            save_path = ROOT / save_path

        with LOCK:
            if _is_running():
                status = {
                    "running": True,
                    "pid": STATE.process.pid if STATE.process else None,
                    "started_at": STATE.started_at,
                    "cmd": STATE.cmd,
                    "log_path": str(STATE.log_path),
                    "model_path": str(STATE.model_path),
                    "last_exit_code": STATE.last_exit_code,
                    "last_error": STATE.last_error,
                    "log_tail": _read_log_tail(STATE.log_path),
                    "model_exists": STATE.model_path.exists(),
                }
                _json_response(
                    self,
                    HTTPStatus.CONFLICT,
                    {"ok": False, "error": "Training is already running", **status},
                )
                return

            save_path.parent.mkdir(parents=True, exist_ok=True)
            STATE.log_path.parent.mkdir(parents=True, exist_ok=True)
            STATE.log_path.write_text("", encoding="utf-8")

            if train_mode == "parallel":
                cmd = [
                    sys.executable,
                    "-u",
                    "parallel_rl_train.py",
                    "--runs",
                    str(max(1, runs)),
                    "--max-parallel",
                    str(max(1, max_parallel)),
                    "--base-seed",
                    str(seed),
                    "--episodes",
                    str(episodes),
                    "--episode-length",
                    str(episode_length),
                    "--eval-every",
                    str(eval_every),
                    "--eval-episodes",
                    str(eval_episodes),
                    "--best-out",
                    str(save_path),
                    "--numpy-threads",
                    str(max(1, numpy_threads)),
                ]
            else:
                cmd = [
                    sys.executable,
                    "-u",
                    "main.py",
                    "--mode",
                    "rltrain",
                    "--episodes",
                    str(episodes),
                    "--episode-length",
                    str(episode_length),
                    "--eval-every",
                    str(eval_every),
                    "--eval-episodes",
                    str(eval_episodes),
                    "--seed",
                    str(seed),
                    "--save-path",
                    str(save_path),
                ]

            log_file = STATE.log_path.open("a", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=0,
            )

            STATE.process = proc
            STATE.started_at = time.time()
            STATE.cmd = cmd
            STATE.model_path = save_path
            STATE.last_exit_code = None
            STATE.last_error = ""

            watcher = threading.Thread(target=_watch_process, args=(proc,), daemon=True)
            watcher.start()

        _json_response(self, HTTPStatus.OK, {"ok": True, "message": "Training started", **_current_status()})

    def _handle_stop(self) -> None:
        with LOCK:
            if not _is_running() or STATE.process is None:
                _json_response(self, HTTPStatus.OK, {"ok": True, "message": "No training process running"})
                return

            proc = STATE.process
            _terminate_process_tree(proc)

        _json_response(self, HTTPStatus.OK, {"ok": True, "message": "Stop signal sent"})

    def _handle_apply(self) -> None:
        body = self._read_json()
        state = body.get("state")
        if not isinstance(state, dict):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Missing state object"})
            return

        model_path_raw = str(body.get("model_path") or str(_current_status().get("model_path") or DEFAULT_MODEL_PATH))
        model_path = Path(model_path_raw)
        if not model_path.is_absolute():
            model_path = ROOT / model_path

        if not model_path.exists():
            _json_response(
                self,
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": f"Model not found: {model_path}"},
            )
            return

        steps = int(body.get("steps", 160))
        candidate_actions = int(body.get("candidate_actions", 1800))

        result = _apply_trained_policy(
            state=state,
            model_path=model_path,
            steps=steps,
            candidate_actions=candidate_actions,
        )

        _json_response(self, HTTPStatus.OK, {"ok": True, **result})

    def log_message(self, format: str, *args: Any) -> None:
        # Keep stdout clean; training logs go to file.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="AI City Planner bridge server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    print(f"Bridge server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
