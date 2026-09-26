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
__all__ = ["Decider", "open_image", "MODEL_ID", "REVISION"]

MODEL_ID = "internlm/Intern-Decision-0.8B"
# _ref.py was vendored from this revision's inference.py; the pin keeps prompt and weights in step.
REVISION = "85a0cc5a99d67ea8d56dfe98115689212867171d"
REF_SHA256 = "62664f5bebb593e825370f82b2733edd13b16118b436d09db2f0c36405b9b4ea"


def open_image(ref: str, root: Path | None) -> Image.Image:
    """Resolve one `images` entry: a base64 data URI, or a local path under `root`.

    URLs are refused on purpose: a local server that fetches URLs is an SSRF hole.
    `root=None` disables paths entirely (the server default).
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
    path = (root / ref).resolve()
    if not path.is_relative_to(root):
        raise ValueError("images: path is outside the allowed directory")
    if not path.is_file():
        raise ValueError(f"images: no such file: {ref}")
    return _decode(path)


def _decode(src) -> Image.Image:
    try:
        with Image.open(src) as im:
            return im.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"images: cannot decode image ({exc})") from exc


class Decider:
    """Load Intern-Decision once, then call `predict(request)` as often as you like.

    request = {"state": ..., "questions": {field: {"type": "noul"|"choice"|"score",
               "instructions": str, "criteria": ...}}, "images": [data-uri or path, ...]}
    """

    def __init__(self, model: str = MODEL_ID, revision: str | None = REVISION,
                 temperature: float | None = None, image_root: str | Path | None = "/"):
        from mlx_vlm import load
        from mlx_vlm.utils import get_model_path

        self.image_root = None if image_root is None else Path(image_root).resolve()
        self.temperature = _ref.validate_temperature(
            _ref.DEFAULT_TEMPERATURE if temperature is None else temperature)
        self.model_id, self.revision = model, revision
        self.full_logits = False

        t0 = time.perf_counter()
        path = get_model_path(model, revision=revision)
        _check_prompt_code(path)
        self.model, self.processor = load(str(path))
        mx.eval(self.model.parameters())
        self.load_ms = (time.perf_counter() - t0) * 1000

        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if _ref.DECISION_TOKEN not in self.tokenizer.get_added_vocab():
            raise ValueError(f"{model} has no {_ref.DECISION_TOKEN} token; not an Intern-Decision checkpoint")
        self.marker_id = self.tokenizer.convert_tokens_to_ids(_ref.DECISION_TOKEN)
        self._symbol_ids: dict[str, int] = {}

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
        images = []
        if row.get("images"):
            parts = []
            for part in messages[1]["content"]:
                if part["type"] == "image_url":
                    images.append(open_image(part["image_url"]["url"], self.image_root))
                    parts.append({"type": "image"})
                else:
                    parts.append(part)
            messages[1]["content"] = parts

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False, add_vision_id=True)
        if images:
            batch = self.processor(text=[text], images=images, return_tensors="np", padding=False)
        else:
            batch = self.tokenizer(text, add_special_tokens=False, return_tensors="np")

        ids = np.asarray(batch["input_ids"])
        positions = np.nonzero(ids[0] == self.marker_id)[0] - 1
        if len(positions) != len(compiled.fields) or (positions < 0).any():
            raise ValueError("decision marker count or position mismatch")
        extra = {k: mx.array(np.asarray(batch[k]))
                 for k in ("pixel_values", "image_grid_thw") if batch.get(k) is not None}
        return compiled, mx.array(ids), positions, extra

    def _logits_at(self, input_ids, idx, extra):
        if not self.full_logits:
            try:
                # Only len(fields) rows are read, so run the 248k-row output head on those
                # rows instead of every token. Same numbers (tests assert it), less work.
                out = self.model(input_ids, skip_logits=True, return_hidden=True, **extra)
                return self.model.language_model.speculative_logits_from_hidden(out.hidden_states[-1][0, idx])
            except (TypeError, AttributeError):
                self.full_logits = True  # mlx-vlm internals moved; fall back, still correct
        out = self.model(input_ids, **extra)
        return getattr(out, "logits", out)[0, idx]

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
    """Warn if the checkpoint ships a different prompt compiler than the one vendored here."""
    ref = Path(path) / "inference.py"
    if ref.is_file() and hashlib.sha256(ref.read_bytes()).hexdigest() != REF_SHA256:
        warnings.warn(
            f"{ref} differs from the prompt compiler vendored in intern_decision_mlx._ref "
            f"(revision {REVISION[:12]}). Decisions may be wrong; pin revision={REVISION!r}.",
            stacklevel=3)
