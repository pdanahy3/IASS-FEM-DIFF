#!/usr/bin/env python3
"""
Stream DDPM denoising steps (decoded 80×80 RGB displacement) for multi-agent-diffusion-mesh.html.

Run from repo root (with optional venv and ``pip install -e ".[train]"``):

  python scripts/diffusion_mesh_infer_server.py --host 127.0.0.1 --port 8765

The browser page POSTs JSON to ``/api/start`` and polls ``/api/session/<id>/next`` for
each step. Images are optional PNG base64 blobs in the JSON body.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import shutil
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from iass_fem_diff.infer.trig_guided_sample import GuidedRunConfig, iter_guided_denoise


def _json_response(handler: BaseHTTPRequestHandler, code: int, obj: dict) -> None:
    body = json.dumps(obj).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _decode_optional_png_b64(data: str | None, tmp: Path, name: str) -> Path | None:
    if not data:
        return None
    raw = base64.b64decode(data)
    p = tmp / name
    p.write_bytes(raw)
    return p


class Session:
    def __init__(self) -> None:
        self.q: queue.Queue[dict] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.done = threading.Event()


SESSIONS: dict[str, Session] = {}


def _run_session(sid: str, body: dict, repo_root: Path) -> None:
    sess = SESSIONS[sid]
    tmp = Path(tempfile.mkdtemp(prefix="mesh_infer_"))
    try:
        ckpt_rel = body.get("checkpoint")
        if not ckpt_rel:
            sess.q.put({"error": "missing checkpoint"})
            return
        ckpt_path = Path(str(ckpt_rel))
        if not ckpt_path.is_absolute():
            ckpt_path = (repo_root / ckpt_path).resolve()
        if not ckpt_path.is_file():
            sess.q.put({"error": f"checkpoint not found: {ckpt_path}"})
            return

        seed_img = _decode_optional_png_b64(body.get("seed_image_png_base64"), tmp, "seed.png")
        goal_img = _decode_optional_png_b64(body.get("goal_image_png_base64"), tmp, "goal.png")

        out_dir = tmp / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = GuidedRunConfig(
            checkpoint_path=ckpt_path,
            out_dir=out_dir,
            seed=int(body.get("seed", 42)),
            steps=int(body.get("steps", 200)),
            device=str(body.get("device", "cuda")),
            seed_image=seed_img,
            strength=float(body.get("strength", 0.35)),
            goal_image=goal_img,
            goal_mix=float(body.get("goal_mix", 0.0)),
            save_every=10**9,
            save_first_last=False,
        )

        for step in iter_guided_denoise(cfg):
            rgb = step.rgb_hwc_u8
            h, w = rgb.shape[0], rgb.shape[1]
            b64 = base64.b64encode(rgb.tobytes()).decode("ascii")
            sess.q.put(
                {
                    "done": False,
                    "step": step.step_i,
                    "t": step.t,
                    "w": w,
                    "h": h,
                    "rgb_b64": b64,
                    "metrics": step.metrics,
                }
            )
        sess.q.put({"done": True})
    except Exception as e:  # noqa: BLE001 — surface errors to client
        sess.q.put({"error": str(e)})
    finally:
        sess.done.set()
        shutil.rmtree(tmp, ignore_errors=True)


def make_handler(repo_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/health":
                _json_response(self, 200, {"ok": True})
                return

            if path.startswith("/api/session/") and path.endswith("/next"):
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    _json_response(self, 404, {"error": "not found"})
                    return
                sid = parts[2]
                sess = SESSIONS.get(sid)
                if not sess:
                    _json_response(self, 404, {"error": "unknown session"})
                    return
                try:
                    item = sess.q.get(timeout=0.25)
                except queue.Empty:
                    _json_response(self, 200, {"pending": True})
                    return
                _json_response(self, 200, item)
                return

            _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/start":
                _json_response(self, 404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                _json_response(self, 400, {"error": "invalid JSON"})
                return

            sid = str(uuid.uuid4())
            sess = Session()
            SESSIONS[sid] = sess
            t = threading.Thread(target=_run_session, args=(sid, body, repo_root), daemon=True)
            sess.thread = t
            t.start()
            _json_response(self, 200, {"session": sid})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream guided denoise steps for multi-agent mesh demo.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Resolve relative checkpoint paths against this directory.",
    )
    args = ap.parse_args()
    handler = make_handler(args.repo_root.resolve())
    httpd = HTTPServer((args.host, args.port), handler)
    print(f"diffusion mesh infer server http://{args.host}:{args.port}  (repo_root={args.repo_root})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("shutdown")


if __name__ == "__main__":
    main()
