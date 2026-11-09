"""Inference and completion-only QLoRA for Qwen3-VL."""

import json
import random
import time
from pathlib import Path

from .data import annotation, fit_image, image_of

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PROMPT = (
    "Read the receipt. Return exactly one JSON object with keys items, subtotal, total. "
    "Each item has keys name, quantity, unit_price, line_total. Include product rows only, "
    "preserve their printed names and order, and keep repeated rows. "
    "Copy printed amounts rather than computing them. Subtotal is the pre-tax or pre-adjustment "
    "subtotal if separately printed; total is the final amount due, not the cash tendered or change. "
    "Use decimal strings without currency symbols or digit grouping, and null for missing values. "
    'Example: {"items":[{"name":"Tea","quantity":"2","unit_price":"5000","line_total":"10000"}],'
    '"subtotal":null,"total":"10000"}. Do not add markdown or explanations.'
)


class VisionModel:
    def __init__(self, model_path=MODEL, adapter=None, max_pixels=524288, *, train=False):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen inference and training require CUDA")
        self.torch = torch
        self.max_pixels = max_pixels
        location = {} if Path(model_path).is_dir() else {"revision": REVISION}
        self.processor = AutoProcessor.from_pretrained(model_path, **location)
        self.processor.tokenizer.padding_side = "left"
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, **location, dtype=torch.bfloat16, device_map={"": 0},
            quantization_config=config, attn_implementation="sdpa",
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        if not train:
            self.model.eval()

    def messages(self, image):
        return [{"role": "user", "content": [
            {"type": "image", "image": fit_image(image, self.max_pixels)},
            {"type": "text", "text": PROMPT},
        ]}]

    def encode(self, messages):
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, padding=True,
            return_dict=True, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        return inputs.to(self.model.device)

    def predict(self, images, max_new_tokens=768):
        inputs = self.encode([self.messages(image) for image in images])
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            )
        return self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )

    def training_input(self, image, gold):
        inputs = self.encode(self.messages(image))
        prompt_length = inputs.input_ids.shape[1]
        completion = json.dumps(gold, ensure_ascii=False, separators=(",", ":")) + "<|im_end|>"
        target = self.processor.tokenizer(
            completion, add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(self.model.device)
        inputs["input_ids"] = self.torch.cat((inputs.input_ids, target), dim=1)
        inputs["attention_mask"] = self.torch.ones_like(inputs.input_ids)
        labels = inputs.input_ids.clone()
        labels[:, :prompt_length] = -100
        inputs["labels"] = labels
        return inputs


def train_adapter(args, data):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(args.seed)
    pipeline = VisionModel(args.model, max_pixels=args.max_pixels, train=True)
    base = prepare_model_for_kbit_training(
        pipeline.model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    base.config.use_cache = False
    pipeline.model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    ))
    pipeline.model.train()
    pipeline.model.print_trainable_parameters()
    params = [p for p in pipeline.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    indices = list(range(len(data)))
    random.Random(args.seed).shuffle(indices)
    if args.steps * args.accum > len(indices):
        raise ValueError("steps × accum exceeds the training split")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    logs = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for idx in indices[step * args.accum:(step + 1) * args.accum]:
            row = data[idx]
            inputs = pipeline.training_input(image_of(row), annotation(row["ground_truth"]))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = pipeline.model(**inputs, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at optimizer step {step + 1}")
            (loss / args.accum).backward()
            loss_total += loss.detach().item() / args.accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        logs.append(loss_total)
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == args.steps:
            print(f"step={step + 1}/{args.steps} mean_loss={loss_total:.4f} "
                  f"elapsed_s={time.perf_counter() - start:.1f}", flush=True)
    pipeline.model.save_pretrained(output)
    result = {
        "model": MODEL, "revision": REVISION, "quantization": "NF4",
        "steps": args.steps, "gradient_accumulation": args.accum,
        "training_receipts": args.steps * args.accum, "seed": args.seed,
        "max_pixels": args.max_pixels, "learning_rate": args.lr,
        "trainable_parameters": sum(p.numel() for p in params),
        "loss_by_step": logs, "training_seconds": time.perf_counter() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "gpu": torch.cuda.get_device_name(),
    }
    (output / "training.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "loss_by_step"}), flush=True)
