"""Small reproducible CPU baseline: OCR detections plus row/amount rules."""

import json
import math
import re
from statistics import median

from .schema import normalize_name, number
from .verification import TextRegion

TOTAL = re.compile(r"^\s*(?:(?:grand|net|nett)\s*)?total(?:\b|(?=\d))", re.I)
SUBTOTAL = re.compile(r"^\s*sub[\s-]*total(?:\b|(?=\d))", re.I)
OTHER = re.compile(r"\b(?:cash|change|paid|tunai|kembali|ppn|service|discount|diskon|tax|vat|"
                   r"payment|receipt|invoice|date|table|cashier)\b", re.I)
AMOUNT = re.compile(r"(?:IDR|Rp\.?)?\s*[+-]?\d+(?:[.,]\d+)*", re.I)


def grouped_regions(detections, width, height, *, deskew=False):
    if not detections:
        return []
    detections = [([[float(x), float(y)] for x, y in box], text)
                  for box, text, _ in detections]
    angle = 0.0
    if deskew:
        slopes = []
        for box, _ in detections:
            dx, dy = box[1][0] - box[0][0], box[1][1] - box[0][1]
            slope = math.atan2(dy, dx)
            if dx >= 10 and abs(slope) <= math.radians(20):
                slopes.append(slope)
        if len(slopes) >= 3:
            candidate = median(slopes)
            if abs(candidate) >= math.radians(1):
                angle = candidate

    cosine, sine = math.cos(angle), math.sin(angle)
    words = []
    for box, text in detections:
        xs, ys = zip(*box, strict=True)
        bounds = (min(xs), min(ys), max(xs), max(ys))
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        if angle:
            row, column = cy * cosine - cx * sine, cx * cosine + cy * sine
            height_hint = math.dist(box[0], box[-1])
        else:
            row, column, height_hint = cy, bounds[0], bounds[3] - bounds[1]
        words.append((row, column, height_hint, text, bounds))
    tolerance = max(3.0, median(word[2] for word in words) * 0.56)
    lines = []
    for word in sorted(words):
        if lines and abs(word[0] - median(w[0] for w in lines[-1])) <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    regions = []
    for line in lines:
        boxes = [word[4] for word in line]
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        coords = tuple(round(max(0, min(1, x)), 5)
                       for x in (x1 / width, y1 / height, x2 / width, y2 / height))
        text = " ".join(word[3] for word in sorted(line, key=lambda w: w[1]))
        regions.append(TextRegion(text, coords))
    return regions


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

    def detect(self, image):
        bgr = self.np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        detections, _ = self.engine(bgr, use_cls=False)
        return detections or []

    def regions(self, image, *, deskew=False):
        return grouped_regions(self.detect(image), image.width, image.height, deskew=deskew)

    def predict(self, images, max_new_tokens=None):
        results = []
        for image in images:
            receipt = extract_lines([region.text for region in self.regions(image)])
            results.append(json.dumps(receipt, ensure_ascii=False))
        return results
