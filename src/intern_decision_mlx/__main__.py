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


def example_request() -> dict:
    req = json.loads((ASSETS / "request.json").read_text())
    req["images"] = [str(ASSETS / "screenshot.png")]
    return req


def _decider(args, image_root="/"):
    from . import Decider
    return Decider(args.model, revision=args.revision, temperature=args.temperature,
                   image_root=image_root)


def cmd_decide(args):
    if args.request is None:
        req = example_request()
    else:
        text = sys.stdin.read() if args.request == "-" else Path(args.request).read_text()
        req = json.loads(text)
    if args.image:
        req["images"] = list(req.get("images") or []) + [str(Path(p).resolve()) for p in args.image]
    print(json.dumps(_decider(args).predict(req), indent=2, ensure_ascii=False))


def cmd_serve(args):
    from . import MODEL_ID, __version__
    root = None if args.allow_paths is None else Path(args.allow_paths).resolve()
    decider = _decider(args, image_root=root)
    health = json.dumps({"status": "ok", "model": args.model or MODEL_ID, "revision": args.revision,
                         "backend": "mlx", "load_ms": round(decider.load_ms), "version": __version__})

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, health)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._send(411, {"error": "Content-Length required"})
            if not 0 < length <= MAX_BODY:
                return self._send(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
            try:
                req = json.loads(self.rfile.read(length))
                return self._send(200, decider.predict(req))
            except (ValueError, TypeError) as exc:  # bad JSON, bad schema, bad image
                return self._send(400, {"error": str(exc)})
            except Exception:
                traceback.print_exc()
                return self._send(500, {"error": "internal error"})

        def log_message(self, fmt, *a):
            sys.stderr.write(f"{self.address_string()} {fmt % a}\n")

    # Single-threaded on purpose: MLX calls are not thread-safe, so requests queue.
    server = HTTPServer((args.host, args.port), Handler)
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


def main(argv=None):
    from . import MODEL_ID, REVISION
    ap = argparse.ArgumentParser(prog="intern-decision-mlx",
                                 description="Screenshot + questions -> calibrated typed decisions, on Apple silicon.")
    ap.add_argument("--model", default=MODEL_ID, help="HF repo id or local path")
    ap.add_argument("--revision", default=REVISION, help="HF revision (pinned to match the vendored prompt code)")
    ap.add_argument("--temperature", type=float, default=None, help="calibration temperature (default: the lab's)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide", help="run one request and print the JSON")
    d.add_argument("request", nargs="?", help="request JSON file, or - for stdin (default: bundled example)")
    d.add_argument("--image", action="append", help="add a local image (repeatable)")
    d.set_defaults(fn=cmd_decide)

    s = sub.add_parser("serve", help="serve POST /v1/systemone on localhost")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--allow-paths", metavar="DIR", help="allow local image paths under DIR (default: data URIs only)")
    s.set_defaults(fn=cmd_serve)

    b = sub.add_parser("bench", help="measure latency and memory on the bundled screenshot")
    b.add_argument("--runs", type=int, default=20)
    b.add_argument("--no-image", action="store_true")
    b.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
