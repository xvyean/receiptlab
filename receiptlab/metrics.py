"""Field metrics with multiset matching so repeated product rows count."""

from collections import Counter
from math import ceil
from statistics import mean

from .schema import ITEM_KEYS

EMPTY = {"items": [], "subtotal": None, "total": None}


def named_rows(receipt):
    return Counter(
        (item["name"].casefold(), item["quantity"], item["unit_price"], item["line_total"])
        for item in receipt["items"]
    )


def per_receipt(gold, parsed, latency):
    pred = parsed if parsed is not None else EMPTY
    target, actual = named_rows(gold), named_rows(pred)
    fields = {}
    for key in ITEM_KEYS:
        def values(receipt):
            return Counter(
                (item["name"].casefold(), item[key] if key != "name" else None)
                for item in receipt["items"] if item[key] is not None
            )

        expected_values, predicted_values = values(gold), values(pred)
        fields[key] = {
            "match": sum((expected_values & predicted_values).values()),
            "expected": expected_values.total(),
            "predicted": predicted_values.total(),
        }
    return {
        "valid": parsed is not None,
        "total_eligible": gold["total"] is not None,
        "total_correct": gold["total"] is not None and pred["total"] == gold["total"],
        "rows_match": sum((target & actual).values()),
        "rows_expected": target.total(),
        "rows_predicted": actual.total(),
        "fields": fields,
        "latency_seconds": latency,
    }


def summarize(records):
    if not records:
        raise ValueError("No examples scored")
    count = len(records)

    def f1(matched, expected, predicted):
        return 2 * matched / (expected + predicted) if expected + predicted else None

    latencies = sorted(r["latency_seconds"] for r in records)
    total_eligible = sum(r["total_eligible"] for r in records)
    total_correct = sum(r["total_correct"] for r in records)
    row_counts = {key: sum(r[key] for r in records)
                  for key in ("rows_match", "rows_expected", "rows_predicted")}
    return {
        "receipts": count,
        "schema_valid_rate": sum(r["valid"] for r in records) / count,
        "total_exact_accuracy": total_correct / total_eligible if total_eligible else None,
        "total_eligible": total_eligible,
        "total_correct": total_correct,
        **row_counts,
        "line_item_exact_f1": f1(*row_counts.values()),
        "field_f1": {
            field: f1(*(sum(r["fields"][field][k] for r in records)
                        for k in ("match", "expected", "predicted")))
            for field in ITEM_KEYS
        },
        "mean_seconds_per_receipt": mean(latencies),
        "p95_seconds_per_receipt": latencies[ceil(0.95 * count) - 1],
    }
