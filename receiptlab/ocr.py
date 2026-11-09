"""Small reproducible CPU baseline: OCR detections plus row/amount rules."""

import json
import re
from statistics import median

from .schema import normalize_name, number

TOTAL = re.compile(r"^\s*(?:(?:grand|net|nett)\s*)?total(?:\b|(?=\d))", re.I)
SUBTOTAL = re.compile(r"^\s*sub[\s-]*total(?:\b|(?=\d))", re.I)
OTHER = re.compile(r"\b(?:cash|change|paid|tunai|kembali|ppn|service|discount|diskon|tax|vat|"
                   r"payment|receipt|invoice|date|table|cashier)\b", re.I)
AMOUNT = re.compile(r"(?:IDR|Rp\.?)?\s*[+-]?\d+(?:[.,]\d+)*", re.I)


def grouped_lines(detections):
    words = []
    for box, text, _ in detections:
        ys = [float(y) for _, y in box]
        words.append((sum(ys) / len(ys), min(float(x) for x, _ in box),
                      max(ys) - min(ys), text))
    if not words:
        return []
    tolerance = max(3.0, median(word[2] for word in words) * 0.56)
    lines = []
    for word in sorted(words):
        if lines and abs(word[0] - median(w[0] for w in lines[-1])) <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [" ".join(w[3] for w in sorted(line, key=lambda w: w[1])) for line in lines]


def extract_lines(lines):
    receipt = {"items": [], "subtotal": None, "total": None}
    for line in lines:
        values = list(AMOUNT.finditer(line))
        if not values:
            continue
        last = values[-1]
        if SUBTOTAL.match(line):
            receipt["subtotal"] = number(last.group())
            continue
        if TOTAL.match(line):
            if not OTHER.search(line):
                receipt["total"] = number(last.group())
            continue
        if receipt["subtotal"] is not None or receipt["total"] is not None or OTHER.search(line):
            continue
        if line[last.end():].strip() or len(values) > 3:
            continue
        first = last
        for candidate in reversed(values[:-1]):
            if line[candidate.end():first.start()].strip(" xX×@"):
                break
            first = candidate
        name = normalize_name(line[:first.start()].strip(" :-x×"))
        if len(name) < 2 or not any(ch.isalpha() for ch in name):
            continue
        suffix = [v for v in values if v.start() >= first.start()]
        receipt["items"].append({
            "name": name,
            "quantity": number(suffix[0].group(), amount=False) if len(suffix) >= 2 else None,
            "unit_price": number(suffix[1].group()) if len(suffix) == 3 else None,
            "line_total": number(last.group()),
        })
    return receipt


class OCR:
    def __init__(self):
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR

        self.np = np
        self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)

    def predict(self, images, max_new_tokens=None):
        results = []
        for image in images:
            bgr = self.np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            detections, _ = self.engine(bgr, use_cls=False)
            results.append(json.dumps(extract_lines(grouped_lines(detections or [])), ensure_ascii=False))
        return results
