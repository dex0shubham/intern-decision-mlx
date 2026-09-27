"""CLI: `intern-decision-mlx decide | serve | bench`."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path

ASSETS = files("intern_decision_mlx") / "assets"
MAX_BODY = 32 * 1024 * 1024  # a few full-resolution screenshots as base64
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def example_request() -> dict:
    req = json.loads((ASSETS / "request.json").read_text())
    req["images"] = [str(ASSETS / "screenshot.png")]
    return req


def _decider(args, image_root="/"):
    from . import Decider
    return Decider(args.model, revision=args.revision, temperature=args.temperature,
                   image_root=image_root, max_side=args.max_side)


def cmd_decide(args):
    if args.request is None:
        req = example_request()
    else:
        text = sys.stdin.read() if args.request == "-" else Path(args.request).read_text()
        req = json.loads(text)
        if not isinstance(req, dict):
            raise ValueError("the request must be a JSON object")
    for p in args.image or []:
        if not Path(p).is_file():  # before the model load, not after it
            raise ValueError(f"no such image: {p}")
    if args.image:
        req["images"] = list(req.get("images") or []) + [str(Path(p).resolve()) for p in args.image]
    print(json.dumps(_decider(args).predict(req), indent=2, ensure_ascii=False))


def make_server(decider, host="127.0.0.1", port=8765, allow_root=None):
    """The HTTP server around a Decider (anything with predict/model_id/revision/load_ms)."""
    from . import __version__
    health = json.dumps({"status": "ok", "model": decider.model_id, "revision": decider.revision,
                         "backend": "mlx", "load_ms": round(decider.load_ms), "version": __version__})
    loopback = host in LOOPBACK

    class Handler(BaseHTTPRequestHandler):
        timeout = 30  # a single-threaded server must not wait forever on an idle or slow client

        def _send(self, code, body):
            data = body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _foreign(self):
            """Block browsers: DNS rebinding (Host) and cross-site requests (Origin)."""
            host_name = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
            origin = self.headers.get("Origin")
            if loopback and host_name not in LOOPBACK:
                return True
            return origin is not None and origin.split("://", 1)[-1].rsplit(":", 1)[0].strip("[]") not in LOOPBACK

        def do_GET(self):
            if self._foreign():
                return self._send(403, {"error": "forbidden"})
            if self.path == "/health":
                return self._send(200, health)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self._foreign():
                return self._send(403, {"error": "forbidden"})
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            # application/json forces a CORS preflight, which this server never grants
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                return self._send(415, {"error": "Content-Type must be application/json"})
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._send(411, {"error": "Content-Length required"})
            if not 0 < length <= MAX_BODY:
                return self._send(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
            try:
                req = json.loads(self.rfile.read(length))
            except (ValueError, RecursionError):
                return self._send(400, {"error": "body is not valid JSON"})
            try:
                return self._send(200, decider.predict(req))
            except (ValueError, RecursionError) as exc:  # schema, limits, images
                return self._send(422, {"error": str(exc)})
            except Exception:
                traceback.print_exc()
                return self._send(500, {"error": "internal error"})

        def log_message(self, fmt, *a):
            sys.stderr.write(f"{self.address_string()} {fmt % a}\n")

    if not loopback:
        print(f"warning: listening on {host} with no authentication; anyone who can reach "
              f"this port can run decisions" + (" and read images under --allow-paths" if allow_root else ""),
              file=sys.stderr)
    # Single-threaded on purpose: MLX calls are not thread-safe, so requests queue.
    return HTTPServer((host, port), Handler)


def cmd_serve(args):
    import mlx.core as mx
    from . import __version__
    root = None if args.allow_paths is None else Path(args.allow_paths).resolve()
    decider = _decider(args, image_root=root)
    mx.set_cache_limit(1 << 30)  # one big request must not pin gigabytes for the process lifetime
    server = make_server(decider, args.host, args.port, root)
    print(f"intern-decision-mlx {__version__}: loaded in {decider.load_ms:.0f} ms, "
          f"POST http://{args.host}:{args.port}/v1/systemone", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_bench(args):
    import mlx.core as mx
    decider = _decider(args)
    req = example_request()
    if args.no_image:
        req.pop("images")

    lean = decider.predict(req)
    decider.full_logits = True
    full = decider.predict(req)
    decider.full_logits = False
    delta = max(abs(lean["answers"][f]["probabilities"][k] - full["answers"][f]["probabilities"][k])
                for f in lean["answers"] for k in lean["answers"][f]["probabilities"])
    assert delta < 1e-6, f"fast readout disagrees with full logits by {delta}"

    for _ in range(3):
        decider.predict(req)  # warm: Metal kernels compile on first use
    mx.reset_peak_memory()
    lat = []
    for _ in range(args.runs):
        t = time.perf_counter()
        decider.predict(req)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    print(json.dumps({
        "input_tokens": lean["usage"]["input_tokens"],
        "fields": len(lean["answers"]),
        "load_ms": round(decider.load_ms),
        "runs": args.runs,
        "p50_ms": round(lat[len(lat) // 2], 1),
        "p95_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1),
        "min_ms": round(lat[0], 1),
        "peak_metal_gib": round(mx.get_peak_memory() / 2**30, 2),
        "lean_vs_full_max_abs_dp": delta,
    }, indent=2))


def _common(model_id, defaults):
    # Subcommands get SUPPRESS defaults, or they would overwrite options given before them.
    dv = (lambda v: v) if defaults else (lambda v: argparse.SUPPRESS)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", default=dv(model_id), help="HF repo id or local path")
    p.add_argument("--revision", default=dv(None), help="HF revision (default: the pinned one for the default model)")
    p.add_argument("--temperature", type=float, default=dv(None), help="calibration temperature (default: the lab's)")
    p.add_argument("--max-side", type=int, default=dv(None), metavar="PX",
                   help="shrink images so the long edge is at most PX (faster; default: no resize)")
    return p


def main(argv=None):
    from . import MODEL_ID, __version__
    common = _common(MODEL_ID, defaults=False)
    ap = argparse.ArgumentParser(prog="intern-decision-mlx", parents=[_common(MODEL_ID, defaults=True)],
                                 description="Screenshot + questions -> calibrated typed decisions, on Apple silicon.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{decide,serve,bench}")

    d = sub.add_parser("decide", parents=[common], help="run one request and print the JSON")
    d.add_argument("request", nargs="?", help="request JSON file, or - for stdin (default: bundled example)")
    d.add_argument("--image", action="append", help="add a local image (repeatable)")
    d.set_defaults(fn=cmd_decide)

    s = sub.add_parser("serve", parents=[common], help="serve POST /v1/systemone on localhost")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--allow-paths", metavar="DIR", help="allow local image paths under DIR (default: data URIs only)")
    s.set_defaults(fn=cmd_serve)

    b = sub.add_parser("bench", parents=[common], help="measure latency and memory on the bundled screenshot")
    b.add_argument("--runs", type=int, default=20)
    b.add_argument("--no-image", action="store_true")
    b.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    if getattr(args, "runs", 1) < 1:
        ap.error("--runs must be at least 1")
    try:
        args.fn(args)
    except (OSError, ValueError) as exc:  # missing files, bad JSON, bad requests, hub/network errors
        ap.exit(1, f"intern-decision-mlx: error: {exc}\n")


if __name__ == "__main__":
    main()
