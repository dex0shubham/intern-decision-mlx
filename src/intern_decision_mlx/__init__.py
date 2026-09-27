"""Intern-Decision on Apple silicon: screenshot + questions -> calibrated typed decisions.

One causal forward over the lab's masked JSON skeleton; logits are read at the position
before each <decision> marker and softmaxed over that field's candidate symbols. No
generate(), no sampling, no KV cache.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import time
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from . import _ref

__version__ = "0.1.0"
__all__ = ["Decider", "open_image", "MODEL_ID", "REVISION", "MAX_TOKENS"]

MODEL_ID = "internlm/Intern-Decision-0.8B"
# _ref.py was vendored from this revision's inference.py; the pin keeps prompt and weights in step.
REVISION = "85a0cc5a99d67ea8d56dfe98115689212867171d"
REF_SHA256 = "62664f5bebb593e825370f82b2733edd13b16118b436d09db2f0c36405b9b4ea"

MAX_TOKENS = 8192                  # the lab's HFBackend max_length: longer inputs are refused
PATCH = 32                         # 16 px patches, 2x2 merged: one token per 32x32 pixels
MAX_PIXELS = MAX_TOKENS * PATCH * PATCH  # an image this size would spend the whole token budget
FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP")  # never EPS/PS: Pillow hands those to Ghostscript


def open_image(ref: str, root: Path | None) -> Image.Image:
    """Resolve one `images` entry: a base64 data URI, or a local path.

    root=None disables paths (the server default). root=DIR resolves relative paths
    against DIR and refuses anything that resolves outside it (symlinks included).
    root=Path("/") means unrestricted: relative paths are relative to the working
    directory, as in the lab's reference. URLs are refused: a local server that
    fetches URLs is an SSRF hole.
    """
    if ref.startswith("data:"):
        header, _, payload = ref.partition(",")
        if not header.endswith(";base64"):
            raise ValueError("images: data URIs must be base64 (data:image/png;base64,...)")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("images: invalid base64 in data URI") from exc
        return _decode(io.BytesIO(raw))
    if "://" in ref:
        raise ValueError("images: URLs are not fetched; send a base64 data URI or a local path")
    if root is None:
        raise ValueError("images: local paths are disabled here; send a base64 data URI")
    base = Path.cwd() if root == Path(root.anchor) else root
    try:
        path = (base / ref).resolve()
        ok = path.is_relative_to(root) and path.is_file()
    except OSError:
        ok = False
    if not ok:
        raise ValueError("images: no such file inside the allowed directory")
    return _decode(path)


def _decode(src) -> Image.Image:
    try:
        with Image.open(src, formats=FORMATS) as im:  # reads the header only
            if im.width * im.height > MAX_PIXELS:
                raise ValueError(f"images: {im.width}x{im.height} is too large; "
                                 f"downscale below {MAX_PIXELS:,} pixels")
            return im.convert("RGB")
    except (OSError, Image.DecompressionBombError):
        raise ValueError("images: not a PNG, JPEG, WEBP, GIF or BMP image") from None


def _visual_tokens(images) -> int:
    return sum(-(-im.width // PATCH) * -(-im.height // PATCH) for im in images)


class Decider:
    """Load Intern-Decision once, then call `predict(request)` as often as you like.

    request = {"state": ..., "questions": {field: {"type": "noul"|"choice"|"score",
               "instructions": str, "criteria": ...}}, "images": [data-uri or path, ...]}
    """

    def __init__(self, model: str = MODEL_ID, revision: str | None = None,
                 temperature: float | None = None, image_root: str | Path | None = "/",
                 max_side: int | None = None):
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
        from mlx_vlm import load
        from mlx_vlm.utils import get_model_path

        if revision is None and model == MODEL_ID:
            revision = REVISION
        self.model_id, self.revision = model, (None if Path(model).exists() else revision)
        self.image_root = None if image_root is None else Path(image_root).resolve()
        # Latency is ~1.2 ms per token and a 1920x1080 image is ~2,200 tokens: shrinking the long
        # edge trades detail for speed. Off by default so outputs match the lab's reference.
        self.max_side = max_side
        self.temperature = _ref.validate_temperature(
            _ref.DEFAULT_TEMPERATURE if temperature is None else temperature)

        t0 = time.perf_counter()
        cached = Path(model).exists() or isinstance(
            try_to_load_from_cache(model, "config.json", revision=revision), str)
        if cached:
            disable_progress_bars()  # keep the bars only for a real download
        try:
            path = get_model_path(model, revision=revision)
        finally:
            if cached:
                enable_progress_bars()
        _check_prompt_code(path)
        self.model, self.processor = load(str(path))
        mx.eval(self.model.parameters())
        self.load_ms = (time.perf_counter() - t0) * 1000

        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        added = self.tokenizer.get_added_vocab()
        if _ref.DECISION_TOKEN not in added:
            raise ValueError(f"{model} has no {_ref.DECISION_TOKEN} token; not an Intern-Decision checkpoint")
        self.marker_id = added[_ref.DECISION_TOKEN]
        self._reserved = tuple(added)  # <|im_start|>, <|image_pad|>, <decision>, ...
        self._symbol_ids: dict[str, int] = {}
        # The fast readout runs the 248k-row output head only on the rows that are read.
        # Decided once: a silent per-call fallback would hide real errors.
        self.full_logits = not hasattr(self.model.language_model, "speculative_logits_from_hidden")

    def _symbol_id(self, symbol: str) -> int:
        if symbol not in self._symbol_ids:
            ids = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"answer symbol {symbol!r} is not a single token")
            self._symbol_ids[symbol] = ids[0]
        return self._symbol_ids[symbol]

    def _encode(self, row):
        compiled = _ref.compile_row(row)
        messages = copy.deepcopy(compiled.messages)
        user = messages[1]["content"]
        user_text = user if isinstance(user, str) else next(p["text"] for p in user if p["type"] == "text")
        smuggled = next((t for t in self._reserved if t in user_text), None)
        if smuggled:  # e.g. <|image_pad|> in page text would silently move the real image
            raise ValueError(f"reserved token {smuggled!r} in state, questions or field names")

        images = []
        if row.get("images"):
            parts = []
            for part in user:
                if part["type"] == "image_url":
                    im = open_image(part["image_url"]["url"], self.image_root)
                    if self.max_side:
                        im.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
                    images.append(im)
                    parts.append({"type": "image"})
                else:
                    parts.append(part)
            messages[1]["content"] = parts
            if _visual_tokens(images) > MAX_TOKENS:
                raise ValueError(f"images: about {_visual_tokens(images):,} image tokens, above {MAX_TOKENS:,}; "
                                 "downscale them (a 1920x1080 screenshot is about 2,000)")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False, add_vision_id=True)
        if images:
            batch = self.processor(text=[text], images=images, return_tensors="np", padding=False)
        else:
            batch = self.tokenizer(text, add_special_tokens=False, return_tensors="np")

        ids = np.asarray(batch["input_ids"])
        if ids.shape[-1] > MAX_TOKENS:
            raise ValueError(f"request is {ids.shape[-1]:,} tokens, above {MAX_TOKENS:,}; "
                             "shorten the state or downscale the images (truncation would move the readout)")
        positions = np.nonzero(ids[0] == self.marker_id)[0] - 1
        if len(positions) != len(compiled.fields) or (positions < 0).any():
            raise ValueError("decision marker count or position mismatch")
        extra = {k: mx.array(np.asarray(batch[k]))
                 for k in ("pixel_values", "image_grid_thw") if batch.get(k) is not None}
        return compiled, mx.array(ids), positions, extra

    def _logits_at(self, input_ids, idx, extra):
        if self.full_logits:
            out = self.model(input_ids, **extra)
            return getattr(out, "logits", out)[0, idx]
        out = self.model(input_ids, skip_logits=True, return_hidden=True, **extra)
        return self.model.language_model.speculative_logits_from_hidden(out.hidden_states[-1][0, idx])

    def predict(self, request: dict) -> dict:
        row = _ref.validate_request(request)
        compiled, input_ids, positions, extra = self._encode(row)
        t0 = time.perf_counter()
        picked = self._logits_at(input_ids, mx.array(positions.tolist()), extra).astype(mx.float32)
        mx.eval(picked)
        inference_ms = (time.perf_counter() - t0) * 1000

        answers = {}
        for i, field in enumerate(compiled.fields):
            question = row["questions"][field]
            options = _ref._options(question)
            values = [v for v, _ in options]
            cand = mx.array([self._symbol_id(s) for s in compiled.symbols[field]])
            probs = dict(zip(values, mx.softmax(picked[i][cand], axis=-1).tolist()))
            best = _ref.argmax(probs)
            answer = {"type": question["type"], "probabilities": probs, "confidence": probs[best]}
            if question["type"] == "noul":
                answer["noul"] = probs["yes"]
            elif question["type"] == "score":
                answer["score"] = sum(float(v) * probs[v] for v in values)
                answer["legend"] = dict(options)
            else:
                answer["choice"] = best
            answers[field] = answer

        result = _ref.scale_result(
            {"answers": answers,
             "usage": {"input_tokens": int(input_ids.shape[-1]), "output_tokens": len(answers)},
             "timing": {"inference_ms": round(inference_ms, 2)}},
            self.temperature)
        for answer in result["answers"].values():
            answer["decision"] = _ref.argmax(answer["probabilities"])
            answer["source"] = "local"
        result["usage"]["decision_count"] = len(answers)
        result["model"] = _ref.MODEL_NAME
        result["backend"] = "mlx"
        return result


def _check_prompt_code(path: Path) -> None:
    """Warn if the checkpoint's prompt compiler is missing or differs from the vendored one."""
    ref = Path(path) / "inference.py"
    if not ref.is_file():
        msg = f"{path} ships no inference.py, so the vendored prompt compiler cannot be checked against it"
    elif hashlib.sha256(ref.read_bytes()).hexdigest() != REF_SHA256:
        msg = (f"{ref} differs from the prompt compiler vendored in intern_decision_mlx._ref "
               f"(revision {REVISION[:12]}); decisions may be wrong")
    else:
        return
    warnings.warn(msg, stacklevel=3)
