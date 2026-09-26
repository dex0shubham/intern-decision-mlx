"""Fast tests run everywhere. Model tests load 1.7 GB: IDMLX_MODEL_TESTS=1 pytest."""
import base64
import importlib.machinery
import io
import json
import os
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from intern_decision_mlx import MODEL_ID, REVISION, _ref, open_image
from intern_decision_mlx.__main__ import example_request

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


def _png_data_uri():
    buf = io.BytesIO()
    Image.new("RGB", (4, 3), (255, 0, 0)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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


def test_open_image_accepts_data_uri_and_rooted_paths(tmp_path):
    assert open_image(_png_data_uri(), None).size == (4, 3)
    Image.new("RGB", (2, 2)).save(tmp_path / "ok.png")
    assert open_image("ok.png", tmp_path.resolve()).size == (2, 2)


@pytest.mark.parametrize("ref", [
    "https://example.com/x.png",          # no fetching: SSRF
    "file:///etc/passwd",
    "data:image/png,notbase64",           # non-base64 data URI
    "data:image/png;base64,@@@",          # invalid base64
    "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
    "../outside.png",                     # path escape
    "/etc/hosts",                         # absolute path outside root
])
def test_open_image_refuses(ref, tmp_path):
    with pytest.raises(ValueError):
        open_image(ref, tmp_path.resolve())


def test_paths_disabled_without_root():
    with pytest.raises(ValueError, match="disabled"):
        open_image("x.png", None)


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
    src ="".join("" if line.startswith("from transformers import") else line
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
