"""Evaluate the payable-total review policy on CORD v2 and external SROIE."""

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

from .__main__ import positive
from .data import REVISION as CORD_REVISION
from .data import annotation, image_of, load_split
from .model import MODEL, REVISION as MODEL_REVISION, VisionModel
from .ocr import OCR, grouped_regions
from .schema import number, parse_prediction
from .verification import appears_anywhere, verify_total

SROIE = "jsdnrs/ICDAR2019-SROIE"
SROIE_REVISION = "bffe40c26759f3376ec2b3ae9031dbba54cd587c"
POLICIES = ("model_has_total", "amount_anywhere", "labeled_total_agreement", "deskewed_total_agreement")


def external_data(shard):
    from datasets import Image as ImageColumn
    from datasets import load_dataset

    if shard:
        data = load_dataset("parquet", data_files={"test": shard}, split="test")
    else:
        data = load_dataset(SROIE, split="test", revision=SROIE_REVISION)
    return data.cast_column("image", ImageColumn(decode=False))


def external_total(row):
    value = row["entities"]["total"]
    if value is None:
        return None
    return number(re.sub(r"^\s*(?:RM|MYR)\s*", "", str(value), flags=re.I))


def wilson(successes, count):
    if count == 0:
        return None
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * count)) / count) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_review(records):
    eligible = [row for row in records if row["gold"] is not None]
    size = len(eligible)
    wrong = sum(row["prediction"] != row["gold"] for row in eligible)
    scale_errors = Counter()
    for row in eligible:
        if row["prediction"] is None or row["prediction"] == row["gold"]:
            continue
        target, prediction = Decimal(row["gold"]), Decimal(row["prediction"])
        for factor in (100, 1000):
            if target and prediction == target * factor:
                scale_errors[str(factor)] += 1
    result = {
        "receipts": len(records),
        "eligible": size,
        "full_accuracy": (size - wrong) / size if size else None,
        "wrong_totals": wrong,
        "exact_scale_errors": dict(scale_errors),
        "ocr_seconds_mean": sum(row["ocr_seconds"] for row in records) / len(records),
        "review_reasons": dict(Counter(row["reason"] for row in records)),
        "policies": {},
    }
    for policy in POLICIES:
        accepted = [row for row in eligible if row["decisions"][policy]]
        correct = sum(row["prediction"] == row["gold"] for row in accepted)
        errors = len(accepted) - correct
        result["policies"][policy] = {
            "accepted": len(accepted),
            "correct": correct,
            "errors": errors,
            "coverage": len(accepted) / size if size else None,
            "precision": correct / len(accepted) if accepted else None,
            "precision_ci95_wilson": wilson(correct, len(accepted)),
            "wrong_routed_to_review": wrong - errors,
        }
    return result


def review_record(index, image, gold, receipt, ocr):
    predicted = None if receipt is None else receipt["total"]
    start = time.perf_counter()
    detections = ocr.detect(image)
    regions = grouped_regions(detections, image.width, image.height)
    aligned = grouped_regions(detections, image.width, image.height, deskew=True)
    elapsed = time.perf_counter() - start
    verified = verify_total(predicted, regions)
    deskewed = verify_total(predicted, aligned)
    return {
        "index": index,
        "gold": gold,
        "prediction": predicted,
        "reason": deskewed["reason"],
        "ocr_seconds": elapsed,
        "decisions": {
            "model_has_total": predicted is not None,
            "amount_anywhere": appears_anywhere(predicted, regions),
            "labeled_total_agreement": verified["status"] == "verified",
            "deskewed_total_agreement": deskewed["status"] == "verified",
        },
    }


def cached_cord(args, ocr):
    data = load_split("test", args.cord_data)
    results = {}
    for cache in args.cached:
        with Path(cache).open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                index, variant = row["index"], row["variant"]
                group = results.setdefault(variant, [])
                if group and index <= group[-1]["index"]:
                    raise ValueError("Each cached variant needs unique indices in ascending order")
                entry = data[index]
                gold = annotation(entry["ground_truth"])["total"]
                if row["gold"]["total"] != gold:
                    raise ValueError(f"Cached CORD label differs at {variant} index {index}")
                group.append(review_record(index, image_of(entry, variant), gold, row["prediction"], ocr))
        print(f"Read {Path(cache).name}: {', '.join(f'{key}={len(value)}' for key, value in results.items())}",
              flush=True)
    return {"dataset": "naver-clova-ix/cord-v2", "revision": CORD_REVISION, "split": "test"}, results


def fresh_sroie(args, ocr):
    data = external_data(args.sroie_shard)
    indices = sorted(random.Random(args.seed).sample(range(len(data)), min(len(data), args.limit)))
    cached = None
    if args.cached:
        cached_rows = [json.loads(line) for line in Path(args.cached).read_text(encoding="utf-8").splitlines()]
        if [row["index"] for row in cached_rows] != indices:
            raise ValueError("Cached SROIE records must match the frozen test indices in order")
        cached = {row["index"]: row for row in cached_rows}
    else:
        if not args.adapter:
            raise ValueError("SROIE inference requires --adapter unless --cached is supplied")
        model = VisionModel(args.model, args.adapter)
        model.predict([image_of(data[indices[0]])], max_new_tokens=32)
    records = []
    for start in range(0, len(indices), args.batch_size):
        batch = indices[start:start + args.batch_size]
        rows = [data[index] for index in batch]
        images = [image_of(row) for row in rows]
        predictions = ([parse_prediction(answer) for answer in model.predict(images, max_new_tokens=args.max_new_tokens)]
                       if cached is None else [{"total": cached[i]["prediction"]} for i in batch])
        for index, row, image, prediction in zip(batch, rows, images, predictions, strict=True):
            gold = external_total(row)
            if cached is not None and cached[index]["gold"] != gold:
                raise ValueError(f"Cached SROIE label differs at index {index}")
            records.append(review_record(index, image, gold, prediction, ocr))
        print(f"SROIE: {len(records)}/{len(indices)}", flush=True)
    return {
        "dataset": SROIE, "revision": SROIE_REVISION, "split": "test",
        "source_size": len(data), "seed": args.seed, "indices": indices,
        "model": MODEL, "model_revision": MODEL_REVISION,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
    }, {"clean": records}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="dataset", required=True)
    cord = sub.add_parser("cord")
    cord.add_argument("--cord-data")
    cord.add_argument("--cached", nargs="+", required=True, help="v0.1 predictions.jsonl files")
    ext = sub.add_parser("sroie")
    ext.add_argument("--sroie-shard", help="Local pinned test parquet")
    ext.add_argument("--model", default=MODEL)
    ext.add_argument("--adapter")
    ext.add_argument("--cached", help="Replay frozen records.jsonl without loading the model")
    ext.add_argument("--seed", type=int, default=42)
    ext.add_argument("--limit", type=positive, default=80)
    ext.add_argument("--batch-size", type=positive, default=8)
    ext.add_argument("--max-new-tokens", type=positive, default=768)
    for cmd in (cord, ext):
        cmd.add_argument("--output", required=True)
    args = cli.parse_args()
    ocr = OCR()
    source, variants = (cached_cord if args.dataset == "cord" else fresh_sroie)(args, ocr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {**source, "variants": {key: summarize_review(value) for key, value in variants.items()}}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "records.jsonl").write_text(
        "".join(json.dumps({"variant": name, **row}) + "\n"
                for name, records in variants.items() for row in records),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
