# Vendored from internlm/Intern-Decision-0.8B, file inference.py,
# revision 85a0cc5a99d67ea8d56dfe98115689212867171d (sha256 62664f5b...b9b4ea).
# Copyright Shanghai AI Laboratory. Licensed under the Apache License, Version 2.0.
#
# Changes (Apache-2.0 s4b): kept only the pure prompt/validation/calibration helpers
# (everything above `class HFBackend`); dropped the torch, transformers and PIL imports;
# rewrote four nested-quote f-strings so the file parses on Python 3.10/3.11.
# The prompt text itself is unchanged: the readout indexes exact token offsets, so any
# drift here silently returns confident wrong answers. tests/ diffs it against the original.
"""Inference-only Intern-Decision: structured candidate scoring using native HF.

All fields share a causal forward over a complete masked assistant skeleton.
Logits immediately before each <decision> marker predict its candidate symbol.
Temperature rescales candidate probabilities; no autoregressive generation occurs.
"""
from __future__ import annotations
import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MODEL_NAME = 'Intern-Decision-0.8B'
DEFAULT_TEMPERATURE = 2.747760550703

DECISION_TOKEN = '<decision>'
ANSWER_SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
SYSTEM_PROMPT = 'You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.'

@dataclass(frozen=True)
class CompiledExample:
    messages: list[dict[str, Any]]
    fields: tuple[str, ...]
    symbols: dict[str, tuple[str, ...]]

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)

def _options(question: Mapping[str, Any]) -> list[tuple[str, str]]:
    kind = question.get('type')
    criteria = question.get('criteria')
    if kind == 'choice':
        if not isinstance(criteria, Mapping):
            raise ValueError('choice criteria must be an object')
        return [(str(key), str(value)) for key, value in criteria.items()]
    if kind == 'score':
        if isinstance(criteria, list):
            return [(str(index), str(value)) for index, value in enumerate(criteria)]
        if isinstance(criteria, Mapping):
            return [(str(key), str(value)) for key, value in criteria.items()]
        raise ValueError('score criteria must be a list or object')
    if kind == 'noul':
        descriptions = criteria if isinstance(criteria, Mapping) else {}
        yes = next((str(descriptions[k]) for k in descriptions if str(k).lower() in {'yes', 'true', '1'}), 'The answer is yes (affirmative, or align with the claim).')
        no = next((str(descriptions[k]) for k in descriptions if str(k).lower() in {'no', 'false', '0'}), 'The answer is no (negative, or disagree with the claim).')
        return [('no', no), ('yes', yes)]
    raise ValueError(f'unsupported question type: {kind!r}')

def _message_content(row: Mapping[str, Any], user_text: str) -> str | list[dict[str, Any]]:
    images = row.get('images') or []
    if not images:
        return user_text
    content: list[dict[str, Any]] = []
    for image in images:
        if isinstance(image, Mapping):
            url = image.get('url') or image.get('path')
            image_wh = image.get('image_wh')
        else:
            url, image_wh = (str(image), None)
        image_url: dict[str, Any] = {'url': url}
        if image_wh is not None:
            image_url['image_wh'] = image_wh
        content.append({'type': 'image_url', 'image_url': image_url})
    content.append({'type': 'text', 'text': user_text})
    return content

def compile_row(row: Mapping[str, Any]) -> CompiledExample:
    questions = row.get('questions')
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError(f"{row.get('id', '<missing id>')}: questions must be a non-empty object")
    fields: list[str] = []
    symbols: dict[str, tuple[str, ...]] = {}
    schema_lines: list[str] = []
    for field, question in questions.items():
        if not isinstance(question, Mapping):
            raise ValueError(f"{row.get('id', '<missing id>')}: question {field!r} is not an object")
        options = _options(question)
        if not options:
            raise ValueError(f"{row.get('id', '<missing id>')}: question {field!r} has no options")
        if len(options) > len(ANSWER_SYMBOLS):
            raise ValueError(f'At most {len(ANSWER_SYMBOLS)} single-token answer symbols are supported')
        field_name = str(field)
        fields.append(field_name)
        field_symbols = tuple(ANSWER_SYMBOLS[:len(options)])
        symbols[field_name] = field_symbols
        schema_lines.append(f"{field_name}: {question.get('instructions', '')}")
        for symbol, (value, description) in zip(field_symbols, options):
            schema_lines.append(f'    {symbol} = {value}: {description}')
    state = _json(row.get('state'))
    user_text = f'Return one answer for every field using the supplied answer symbols.\n\n## State\n{state}\n## Decision schema\n' + '\n'.join(schema_lines)
    if DECISION_TOKEN in user_text:
        raise ValueError('Reserved decision marker appears in input evidence')
    skeleton = json.dumps(dict.fromkeys(fields, DECISION_TOKEN), ensure_ascii=False, indent=4)
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': _message_content(row, user_text)}, {'role': 'assistant', 'content': skeleton}]
    return CompiledExample(messages=messages, fields=tuple(fields), symbols=symbols)

def validate_request(request):
    if not isinstance(request, dict) or 'state' not in request:
        raise ValueError('Supply a JSON object containing state and questions.')
    images = request.get('images') or []
    if images:
        if not isinstance(images, list) or not 1 <= len(images) <= 8:
            raise ValueError('Upload 1–8 images.')
        if not all((isinstance(image, str) and bool(image) for image in images)):
            raise ValueError('Images must be nonempty local file paths.')
    questions = request.get('questions')
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 16:
        raise ValueError('Supply 1–16 questions.')
    clean = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Question names must be nonempty strings.')
        if not isinstance(question, dict) or question.get('type') not in {'choice', 'score', 'noul'}:
            raise ValueError('Question type must be choice, score, or noul.')
        kind = question['type']
        criteria = question.get('criteria')
        if kind == 'choice' and (not isinstance(criteria, dict)):
            raise ValueError('Choice criteria must be an object.')
        if kind == 'score':
            if not isinstance(criteria, (list, dict)):
                raise ValueError('Score criteria must be a list or object.')
            if isinstance(criteria, dict):
                try:
                    if not all((math.isfinite(float(key)) for key in criteria)):
                        raise ValueError
                except (ValueError, TypeError) as exc:
                    raise ValueError('Score option keys must be finite numbers.') from exc
        if kind != 'noul' and (not 1 <= len(criteria) <= len(ANSWER_SYMBOLS)):
            raise ValueError(f'Supply 1–{len(ANSWER_SYMBOLS)} options.')
        clean[name] = {key: copy.deepcopy(question[key]) for key in ('type', 'instructions', 'criteria') if key in question}
    row = {'state': copy.deepcopy(request['state']), 'questions': clean}
    if images:
        row['images'] = list(images)
    compile_row(row)
    return row

def argmax(probs):
    return min(probs, key=lambda label: (-probs[label], label))

def validate_temperature(temperature):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    return temperature

def scale_probabilities(probs, temperature):
    temperature = validate_temperature(temperature)
    if not probs or any((not math.isfinite(p) or p < 0 or p > 1 for p in probs.values())):
        raise ValueError('Invalid probability distribution')
    if abs(sum(probs.values()) - 1.0) > 0.001:
        raise ValueError('Probabilities must sum to one')
    if temperature == 1.0:
        return dict(probs)
    logs = {label: math.log(p) if p else -math.inf for label, p in probs.items()}
    maximum = max(logs.values())
    weights = {label: math.exp((logp - maximum) / temperature) for label, logp in logs.items()}
    total = sum(weights.values())
    result = {label: p / total for label, p in weights.items()}
    if argmax(result) != argmax(probs):
        raise ArithmeticError('Floating-point temperature scaling changed argmax')
    return result

def scale_result(result, temperature):
    """Scale all fields consistently, including API confidence/noul/score values."""
    scaled = copy.deepcopy(result)
    for answer in scaled['answers'].values():
        probs = scale_probabilities(answer['probabilities'], temperature)
        answer['probabilities'] = probs
        best = argmax(probs)
        answer['confidence'] = probs[best]
        if answer['type'] == 'choice':
            answer['choice'] = best
        elif answer['type'] == 'noul':
            answer['noul'] = probs['yes']
        elif answer['type'] == 'score':
            answer['score'] = sum((float(label) * prob for label, prob in probs.items()))
        else:
            raise ValueError('Unknown answer type')
    scaled['calibration'] = {'method': 'temperature-scaling', 'temperature': validate_temperature(temperature)}
    return scaled

