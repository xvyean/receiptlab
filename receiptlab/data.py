"""Read official CORD v2 splits without mixing copies across splits."""

import io
import json
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from .schema import normalize_name, number

DATASET = "naver-clova-ix/cord-v2"
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
SPLITS = {"train": 4, "validation": 1, "test": 1}


def load_split(split, local=None):
    from datasets import Image as ImageColumn
    from datasets import load_dataset

    if local:
        files = sorted(Path(local).glob(f"{split}-*.parquet"))
        if len(files) != SPLITS[split]:
            raise ValueError(f"Expected {SPLITS[split]} Parquet files for {split} in {local}")
        data = load_dataset("parquet", data_files={split: [str(f) for f in files]}, split=split)
    else:
        data = load_dataset(DATASET, split=split, revision=REVISION)
    return data.cast_column("image", ImageColumn(decode=False))


def annotation(raw):
    doc = json.loads(raw)["gt_parse"] if isinstance(raw, str) else raw["gt_parse"]
    menu = doc.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    items = []
    for row in menu:
        name = normalize_name(str(row.get("nm") or ""))
        if not name:
            continue
        items.append({
            "name": name,
            "quantity": number(row.get("cnt"), amount=False),
            "unit_price": number(row.get("unitprice")),
            "line_total": number(row.get("itemsubtotal") or row.get("price")),
        })
    return {
        "items": items,
        "subtotal": number(doc.get("sub_total", {}).get("subtotal_price")),
        "total": number(doc.get("total", {}).get("total_price")),
    }


def image_of(row, variant="clean"):
    data = row["image"]
    with Image.open(io.BytesIO(data["bytes"]) if data.get("bytes") else data["path"]) as image:
        clean = ImageOps.exif_transpose(image).convert("RGB")
    if variant == "blur":
        return clean.filter(ImageFilter.GaussianBlur(radius=2.0))
    if variant == "tilt":
        return clean.rotate(9, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    return clean


def fit_image(image, max_pixels):
    image = image.convert("RGB")
    scale = min(1.0, (max_pixels / (image.width * image.height)) ** 0.5)
    if scale < 1.0:
        size = max(32, int(image.width * scale)), max(32, int(image.height * scale))
        return image.resize(size, Image.Resampling.LANCZOS)
    return image
