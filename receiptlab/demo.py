"""Local browser demo. Start with python -m receiptlab.demo."""

import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from .__main__ import engines, make_engine
from .schema import checks, parse_prediction, to_csv

MAX_UPLOAD = 12 * 1024 * 1024
MAX_PIXELS = 16_000_000


def example_png():
    """Generate a demo receipt with fictional products and no personal data."""
    image = Image.new("RGB", (705, 690), "#fffdf3")
    draw = ImageDraw.Draw(image)
    heading = ImageFont.load_default(size=43)
    font = ImageFont.load_default(size=27)
    draw.text((45, 42), "DEMO MARKET", fill="#171717", font=heading)
    draw.text((45, 105), "SAMPLE RECEIPT / IDR", fill="#303030", font=font)
    draw.line((45, 160, 660, 160), fill="#454545", width=2)
    cols = (45, 335, 430, 565)
    for y, values in [
        (187, ("ITEM", "QTY", "UNIT", "TOTAL")),
        (248, ("Green Tea", "2", "5000", "10000")),
        (305, ("Rice Bowl", "1", "25000", "25000")),
        (362, ("Orange Juice", "1", "12000", "12000")),
    ]:
        for x, value in zip(cols, values, strict=True):
            draw.text((x, y), value, fill="#171717", font=font)
    draw.line((45, 431, 660, 431), fill="#454545", width=2)
    for y, label, amount in [
        (466, "SUBTOTAL", "47000"),
        (518, "SERVICE", "2000"),
        (570, "TOTAL", "49000"),
    ]:
        draw.text((45, y), label, fill="#171717", font=font)
        draw.text((565, y), amount, fill="#171717", font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def serve(args):
    engine = make_engine(args)
    lock = threading.Lock()
    html = Path(__file__).with_name("demo.html").read_bytes()
    example = example_png()

    class Request(BaseHTTPRequestHandler):
        def send(self, content, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def json(self, payload, status=200):
            self.send(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      "application/json; charset=utf-8", status)

        def do_GET(self):
            if self.path == "/":
                self.send(html, "text/html; charset=utf-8")
            elif self.path == "/example.png":
                self.send(example, "image/png")
            elif self.path == "/health":
                self.json({"engine": args.engine, "adapter": bool(args.adapter)})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path not in {"/api/extract", "/api/review", "/api/csv"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_UPLOAD:
                    self.json({"error": "请求应小于 12 MB。"}, 413)
                    return
                body = self.rfile.read(length)
                if self.path == "/api/extract":
                    with Image.open(io.BytesIO(body)) as raw_image:
                        if raw_image.width * raw_image.height > MAX_PIXELS:
                            self.json({"error": "图片像素过多，请缩小后重试。"}, 413)
                            return
                        image = ImageOps.exif_transpose(raw_image).convert("RGB")
                    with lock:
                        raw = engine.predict([image], max_new_tokens=args.max_new_tokens)[0]
                    receipt = parse_prediction(raw)
                else:
                    receipt = parse_prediction(body)
                if receipt is None:
                    self.json({"error": "无法解析为票据格式，请核对原图或输入。"}, 422)
                elif self.path == "/api/csv":
                    self.send(to_csv(receipt).encode("utf-8-sig"), "text/csv; charset=utf-8")
                else:
                    self.json({"receipt": receipt, "review": checks(receipt)})
            except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
                self.json({"error": "无法读取输入。请使用 PNG、JPEG 或有效的票据 JSON。"}, 400)
            except Exception:
                import traceback
                traceback.print_exc()
                self.json({"error": "识别服务遇到错误，请查看服务日志。"}, 500)

    with ThreadingHTTPServer(("127.0.0.1", args.port), Request) as server:
        print(f"ReceiptLab on http://127.0.0.1:{args.port} ({args.engine})", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    engines(parser)
    parser.add_argument("--port", type=int, default=7860)
    serve(parser.parse_args())
