"""Canonical fields and deterministic receipt checks."""

import csv
import io
import json
import re
import unicodedata
from decimal import Decimal

ITEM_KEYS = ("name", "quantity", "unit_price", "line_total")
HEADER_KEYS = ("subtotal", "total")


def number(value, *, amount=True):
    """Return a decimal string; accept common grouping on Indonesian receipts."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (float, int, Decimal)):
        text = str(value)
    else:
        text = unicodedata.normalize("NFKC", str(value)).strip()
        text = re.sub(r"^(?:IDR|RP\.?)\s*", "", text, flags=re.I)
        text = re.sub(r"\s+", "", text)
    if not re.fullmatch(r"[+-]?\d+(?:[.,]\d+)*", text):
        return None
    if isinstance(value, str):
        if "," in text and "." in text:
            decimal = "," if text.rfind(",") > text.rfind(".") else "."
            text = text.replace("." if decimal == "," else ",", "")
            text = text.replace(decimal, ".")
        elif "," in text or "." in text:
            sep = "," if "," in text else "."
            groups = text.lstrip("+-").split(sep)
            if (amount or len(groups) > 2) and all(len(x) == 3 for x in groups[1:]):
                text = text.replace(sep, "")
            elif len(groups) == 2:
                text = text.replace(sep, ".")
            else:
                return None
    parsed = Decimal(text)
    if not parsed:
        return "0"
    formatted = format(parsed, "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def normalize_name(value):
    return " ".join(unicodedata.normalize("NFKC", value).split())


def parse_prediction(raw):
    """Accept only schema-shaped JSON. Do not silently repair malformed output."""
    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj

    try:
        doc = json.loads(raw, object_pairs_hook=unique_keys)
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(doc, dict) or set(doc) != {"items", *HEADER_KEYS}:
        return None
    if not isinstance(doc["items"], list) or len(doc["items"]) > 250:
        return None

    def parse_number(value, *, amount=True):
        if value is None:
            return None
        if isinstance(value, str) and not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            raise ValueError("Expected a decimal string without grouping")
        if type(value) not in {str, int, float}:
            raise ValueError("Expected a decimal number")
        result = number(Decimal(str(value)), amount=amount)
        if result is None:
            raise ValueError("Expected a finite decimal number")
        return result

    items = []
    try:
        for item in doc["items"]:
            if not isinstance(item, dict) or set(item) != set(ITEM_KEYS):
                return None
            name = item["name"]
            if not isinstance(name, str) or not normalize_name(name):
                return None
            items.append({
                "name": normalize_name(name),
                "quantity": parse_number(item["quantity"], amount=False),
                "unit_price": parse_number(item["unit_price"]),
                "line_total": parse_number(item["line_total"]),
            })
        return {
            "items": items,
            **{key: parse_number(doc[key]) for key in HEADER_KEYS},
        }
    except (ValueError, ArithmeticError):
        return None


def checks(receipt):
    """Flag differences for review, since discounts and rounding can be legitimate."""
    issues = []
    for idx, row in enumerate(receipt["items"], start=1):
        if any(row[k] is None for k in ITEM_KEYS[1:]):
            continue
        qty, unit, total = (Decimal(row[k]) for k in ITEM_KEYS[1:])
        if abs(qty * unit - total) > Decimal("0.01"):
            issues.append({"row": idx, "kind": "line_amount_mismatch",
                           "expected": str(qty * unit), "observed": str(total)})
    values = [row["line_total"] for row in receipt["items"]]
    if values and all(v is not None for v in values) and receipt["subtotal"] is not None:
        summed = sum(Decimal(value) for value in values)
        if abs(summed - Decimal(receipt["subtotal"])) > Decimal("0.01"):
            issues.append({"row": None, "kind": "subtotal_mismatch",
                           "expected": str(summed), "observed": receipt["subtotal"]})
    return issues


def to_csv(receipt):
    def safe_cell(value):
        # Prevent item names from being interpreted as spreadsheet formulas.
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=ITEM_KEYS)
    writer.writeheader()
    for row in receipt["items"]:
        writer.writerow({**row, "name": safe_cell(row["name"])})
    return out.getvalue()
