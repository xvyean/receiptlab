"""Run dataset evaluation, inference or single-GPU training."""

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageOps

from .data import DATASET, REVISION, annotation, image_of, load_split
from .metrics import per_receipt, summarize
from .model import MODEL, VisionModel, train_adapter
from .ocr import OCR
from .schema import checks, parse_prediction, to_csv


def make_engine(args):
    if args.engine == "ocr":
        if args.adapter:
            raise ValueError("--adapter is only used with the Qwen engine")
        return OCR()
    return VisionModel(args.model, args.adapter, args.max_pixels)


def evaluate(args):
    data = load_split(args.split, args.data)
    end = min(len(data), args.limit or len(data))
    engine = make_engine(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    variants = args.variants
    records = {variant: [] for variant in variants}
    engine.predict([image_of(data[0])], max_new_tokens=min(64, args.max_new_tokens))
    if args.engine == "qwen":
        engine.torch.cuda.reset_peak_memory_stats()
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        for variant in variants:
            for start in range(0, end, args.batch_size):
                rows = [data[i] for i in range(start, min(end, start + args.batch_size))]
                images = [image_of(row, variant) for row in rows]
                t0 = time.perf_counter()
                answers = engine.predict(images, max_new_tokens=args.max_new_tokens)
                seconds = (time.perf_counter() - t0) / len(rows)
                for offset, (row, raw) in enumerate(zip(rows, answers, strict=True)):
                    gold = annotation(row["ground_truth"])
                    parsed = parse_prediction(raw)
                    scored = per_receipt(gold, parsed, seconds)
                    records[variant].append(scored)
                    predictions.write(json.dumps({
                        "index": start + offset, "variant": variant, "gold": gold,
                        "prediction": parsed, "raw": raw, **scored,
                    }, ensure_ascii=False) + "\n")
                predictions.flush()
                done = start + len(rows)
                if done == end or start == 0 or done % (args.batch_size * 5) == 0:
                    print(f"{variant}: {done}/{end}", flush=True)
    summary = {
        "dataset": DATASET, "dataset_revision": REVISION,
        "split": args.split, "engine": args.engine, "model": MODEL if args.engine == "qwen" else None,
        "adapter": bool(args.adapter), "batch_size": args.batch_size,
        "max_pixels": args.max_pixels if args.engine == "qwen" else None,
        "max_new_tokens": args.max_new_tokens if args.engine == "qwen" else None,
        "device": engine.torch.cuda.get_device_name() if args.engine == "qwen" else "CPU",
        "peak_allocated_mib": engine.torch.cuda.max_memory_allocated() / 2**20
        if args.engine == "qwen" else None,
        "variants": {variant: summarize(values) for variant, values in records.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def infer(args):
    with Image.open(args.image) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    raw = make_engine(args).predict([image], max_new_tokens=args.max_new_tokens)[0]
    receipt = parse_prediction(raw)
    if receipt is None:
        raise SystemExit("Model did not return valid receipt JSON:\n" + raw)
    if args.csv:
        Path(args.csv).write_text(to_csv(receipt), encoding="utf-8-sig")
    print(json.dumps({"receipt": receipt, "review": checks(receipt)}, ensure_ascii=False, indent=2))


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def engines(parser):
    parser.add_argument("--engine", choices=("ocr", "qwen"), default="qwen")
    parser.add_argument("--model", default=MODEL, help="Pinned Hub model or local model directory")
    parser.add_argument("--adapter", help="Local QLoRA adapter directory")
    parser.add_argument("--max-pixels", type=positive, default=524288)
    parser.add_argument("--max-new-tokens", type=positive, default=768)


def main():
    cli = argparse.ArgumentParser(prog="python -m receiptlab")
    subs = cli.add_subparsers(dest="command", required=True)
    ev = subs.add_parser("evaluate")
    engines(ev)
    ev.add_argument("--data", help="Directory with all CORD Parquet shards")
    ev.add_argument("--split", choices=("validation", "test"), default="validation")
    ev.add_argument("--variants", nargs="+", choices=("clean", "blur", "tilt"), default=["clean"])
    ev.add_argument("--limit", type=positive)
    ev.add_argument("--batch-size", type=positive, default=1)
    ev.add_argument("--output", required=True)
    ev.set_defaults(run=evaluate)
    one = subs.add_parser("infer")
    engines(one)
    one.add_argument("image")
    one.add_argument("--csv")
    one.set_defaults(run=infer)
    train = subs.add_parser("train")
    train.add_argument("--data")
    train.add_argument("--model", default=MODEL)
    train.add_argument("--output", required=True)
    train.add_argument("--steps", type=positive, default=60)
    train.add_argument("--accum", type=positive, default=4)
    train.add_argument("--max-pixels", type=positive, default=524288)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(run=lambda a: train_adapter(a, load_split("train", a.data)))
    args = cli.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
