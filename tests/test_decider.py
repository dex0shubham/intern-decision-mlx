"""Fast tests run everywhere. Model tests load 1.7 GB: IDMLX_MODEL_TESTS=1 pytest."""
import base64
import http.client
import importlib.machinery
import io
import json
import os
import sys
import threading
import types
from pathlib import Path

import pytest
from PIL import Image

from intern_decision_mlx import MAX_PIXELS, MAX_TOKENS, MODEL_ID, REVISION, _ref, open_image
from intern_decision_mlx.__main__ import example_request, make_server

GROUNDED = {
    "state": "Answer using only what is visible in the screenshot.",
    "questions": {
        "left_button_color": {"type": "choice", "instructions": "What is the background color of the button on the LEFT?",
                              "criteria": {"red": "red", "green": "green", "blue": "blue", "grey": "grey/white"}},
        "has_error": {"type": "noul", "instructions": "Does the screen show an error message?"},
        "button_count": {"type": "choice", "instructions": "How many clickable buttons are in the dialog?",
                         "criteria": {"one": "1", "two": "2", "three": "3", "four": "4"}},
    },
}
GROUNDED_TRUTH = {"left_button_color": "red", "has_error": "yes", "button_count": "two"}

REQUESTS = [
    example_request(),
    dict(GROUNDED),
    {"state": {"ticket": "Mi pedido llegó roto 😞", "tier": "gold"},
     "questions": {"route": {"type": "choice", "instructions": "Queue?", "criteria": {"billing": "b", "returns": "r"}},
                   "urgency": {"type": "score", "instructions": "Urgency", "criteria": {"1": "low", "2": "mid", "5": "high"}}},
     "images": ["a.png", "b.png"]},
]


def _data_uri(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def _png(size, mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format="PNG")
    return buf.getvalue()


EPS = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 32 32\nnewpath 0 0 moveto 32 32 lineto stroke\nshowpage\n"


# ---- fast: no model -------------------------------------------------------------------

def test_one_marker_per_field():
    compiled = _ref.compile_row(_ref.validate_request(example_request()))
    skeleton = json.loads(compiled.messages[2]["content"])
    assert list(skeleton) == list(compiled.fields) == ["screen", "has_error", "next_click"]
    assert set(skeleton.values()) == {_ref.DECISION_TOKEN}
    assert compiled.symbols["screen"] == ("A", "B", "C", "D")


def test_request_validation():
    too_many = {"state": "", "questions": {f"q{i}": {"type": "noul"} for i in range(17)}}
    for bad in [{}, {"state": ""}, too_many,
                {"state": "", "questions": {"q": {"type": "maybe"}}},
                {"state": "", "questions": {"q": {"type": "choice", "criteria": ["a"]}}},
                {"state": "", "questions": {"q": {"type": "noul"}}, "images": [""]},
                {"state": "<decision>", "questions": {"q": {"type": "noul"}}}]:
        with pytest.raises(ValueError):
            _ref.validate_request(bad)


def test_calibration_keeps_argmax_and_mass():
    for probs in [{"no": 0.3, "yes": 0.7}, {"A": 0.5, "B": 0.49, "C": 0.01}, {"x": 1.0, "y": 0.0}]:
        scaled = _ref.scale_probabilities(probs, _ref.DEFAULT_TEMPERATURE)
        assert _ref.argmax(scaled) == _ref.argmax(probs)
        assert abs(sum(scaled.values()) - 1) < 1e-9
        assert max(scaled.values()) <= max(probs.values()) + 1e-12  # T > 1 only softens


def test_open_image_accepts_data_uri_and_paths(tmp_path, monkeypatch):
    assert open_image(_data_uri(_png((4, 3))), None).size == (4, 3)
    Image.new("RGB", (2, 2)).save(tmp_path / "ok.png")
    assert open_image("ok.png", tmp_path.resolve()).size == (2, 2)          # --allow-paths DIR
    monkeypatch.chdir(tmp_path)
    assert open_image("ok.png", Path("/")).size == (2, 2)                   # unrestricted: relative to CWD
    assert open_image(str(tmp_path / "ok.png"), Path("/")).size == (2, 2)


@pytest.mark.parametrize("ref", [
    "https://example.com/x.png",          # no fetching: SSRF
    "file:///etc/passwd",
    "data:image/png,notbase64",           # non-base64 data URI
    "data:image/png;base64,@@@",          # invalid base64
    _data_uri(b"not an image"),
    _data_uri(EPS),                       # PostScript must never reach Ghostscript
    "../outside.png",                     # path escape
    "/etc/hosts",                         # absolute path outside root
    "link.png",                           # symlink inside root pointing outside
])
def test_open_image_refuses(ref, tmp_path):
    (tmp_path / "link.png").symlink_to("/etc/hosts")
    with pytest.raises(ValueError):
        open_image(ref, tmp_path.resolve())


def test_paths_disabled_without_root():
    with pytest.raises(ValueError, match="disabled"):
        open_image("x.png", None)


def test_oversized_image_refused_before_decoding():
    assert 4096 * 2100 > MAX_PIXELS
    with pytest.raises(ValueError, match="too large"):   # ~1 KB on the wire, 8.6M pixels
        open_image(_data_uri(_png((4096, 2100), mode="1")), None)


def test_vendored_prompt_code_matches_the_lab():
    """The readout reads exact token offsets: any drift in _ref returns confident garbage."""
    hub = pytest.importorskip("huggingface_hub")
    original = hub.try_to_load_from_cache(MODEL_ID, "inference.py", revision=REVISION)
    if not isinstance(original, str):
        pytest.skip("original inference.py not in the HF cache")
    if sys.version_info < (3, 12):
        pytest.skip("the lab's file uses Python 3.12 f-string syntax")
    stub = types.ModuleType("torch")
    stub.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    stub.inference_mode = lambda: (lambda f: f)  # used as a decorator on the (unused) HF backend
    src = "".join("" if line.startswith("from transformers import") else line
                  for line in Path(original).read_text().splitlines(keepends=True))
    lab = types.ModuleType("lab_inference")
    saved = sys.modules.get("torch")
    sys.modules["torch"], sys.modules["lab_inference"] = stub, lab
    try:
        exec(compile(src, original, "exec"), lab.__dict__)
    finally:
        sys.modules.pop("lab_inference")
        if saved is None:
            sys.modules.pop("torch")
        else:
            sys.modules["torch"] = saved
    for req in REQUESTS:
        mine, theirs = _ref.compile_row(_ref.validate_request(req)), lab.compile_row(lab.validate_request(req))
        assert (mine.messages, mine.fields, mine.symbols) == (theirs.messages, theirs.fields, theirs.symbols)
    for name in ("SYSTEM_PROMPT", "DECISION_TOKEN", "ANSWER_SYMBOLS", "DEFAULT_TEMPERATURE"):
        assert getattr(_ref, name) == getattr(lab, name)


class FakeDecider:
    model_id, revision, load_ms = "fake", None, 1.0

    def predict(self, req):
        if req.get("state") == "bad":
            raise ValueError("bad request")
        return {"answers": {}, "echo": req["state"]}


@pytest.fixture(scope="module")
def server():
    srv = make_server(FakeDecider(), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def _call(port, method="POST", path="/v1/systemone", body=b'{"state": "ok"}', headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Content-Type": "application/json"} if headers is None else headers
    conn.request(method, path, body=body if method == "POST" else None, headers=h)
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_server_happy_path(server):
    assert _call(server)[0] == 200
    assert _call(server, "GET", "/health")[0] == 200
    assert _call(server, headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"})[0] == 200


@pytest.mark.parametrize("kwargs,status", [
    ({"headers": {"Content-Type": "text/plain"}}, 415),                                         # browser 'simple' POST
    ({"headers": {"Content-Type": "application/json", "Host": "evil.example:8765"}}, 403),     # DNS rebinding
    ({"headers": {"Content-Type": "application/json", "Origin": "https://evil.example"}}, 403),
    ({"method": "GET", "path": "/health", "headers": {"Host": "evil.example"}}, 403),
    ({"body": b"{nope"}, 400),
    ({"body": b"[" * 100000 + b"]" * 100000}, 400),                                             # JSON recursion bomb
    ({"body": b'{"state": "bad"}'}, 422),
    ({"path": "/v1/other"}, 404),
])
def test_server_refusals(server, kwargs, status):
    assert _call(server, **kwargs)[0] == status


# ---- slow: loads the model ------------------------------------------------------------

needs_model = pytest.mark.skipif(os.environ.get("IDMLX_MODEL_TESTS") != "1",
                                 reason="set IDMLX_MODEL_TESTS=1 to load the 1.7 GB model")


@pytest.fixture(scope="module")
def decider():
    from intern_decision_mlx import Decider
    return Decider()


@needs_model
def test_bundled_example_answers(decider):
    """Also the regression check: same weights + same prompt must give the same answers."""
    req = example_request()
    out = decider.predict(req)
    a = out["answers"]
    assert list(a) == list(req["questions"])
    for ans in a.values():
        assert abs(sum(ans["probabilities"].values()) - 1) < 1e-3
        assert ans["decision"] == _ref.argmax(ans["probabilities"])
    assert {f: ans["decision"] for f, ans in a.items()} == {
        "screen": "payment_error", "has_error": "yes", "next_click": "contact_support"}
    assert a["has_error"]["noul"] == a["has_error"]["probabilities"]["yes"]
    assert out["calibration"]["temperature"] == _ref.DEFAULT_TEMPERATURE


@needs_model
def test_the_image_is_actually_read(decider):
    image = example_request()["images"]
    seen = decider.predict(dict(GROUNDED, images=image))["answers"]
    blind = decider.predict(dict(GROUNDED))["answers"]
    assert any(seen[f]["probabilities"] != blind[f]["probabilities"] for f in GROUNDED_TRUTH)
    assert {f: seen[f]["decision"] for f in GROUNDED_TRUTH} == GROUNDED_TRUTH


@needs_model
def test_fast_readout_equals_full_logits(decider):
    assert decider.full_logits is False, "fast readout unavailable: this test would pass vacuously"
    req = example_request()
    lean = decider.predict(req)
    decider.full_logits = True
    try:
        full = decider.predict(req)
    finally:
        decider.full_logits = False
    for f, ans in lean["answers"].items():
        for k, p in ans["probabilities"].items():
            assert abs(p - full["answers"][f]["probabilities"][k]) < 1e-6


@needs_model
def test_limits_and_reserved_tokens(decider):
    q = {"q": {"type": "noul", "instructions": "Is it?"}}
    with pytest.raises(ValueError, match="8,192"):
        decider.predict({"state": "word " * (MAX_TOKENS + 500), "questions": q})
    with pytest.raises(ValueError, match="image tokens"):
        decider.predict({"state": "x", "questions": q, "images": [_data_uri(_png((1920, 1080)))] * 8})
    for token in ("<|image_pad|>", "<|im_start|>", "<|vision_start|>"):
        with pytest.raises(ValueError, match="reserved token"):
            decider.predict({"state": f"page text {token}", "questions": q})


@needs_model
def test_max_side_shrinks_images(decider):
    full = decider.predict(example_request())["usage"]["input_tokens"]
    decider.max_side = 256
    try:
        small = decider.predict(example_request())["usage"]["input_tokens"]
    finally:
        decider.max_side = None
    assert small < full
