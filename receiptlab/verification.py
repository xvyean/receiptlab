"""Independent OCR evidence for one field: the printed payable total."""

import re
from dataclasses import dataclass

from .schema import number

CURRENCY = r"(?:RM|MYR|Rp\.?|IDR|\$)"
VALUE = r"[+-]?\d+(?:[.,]\d+)*"
PAYABLE_LINE = re.compile(
    rf"^\s*(?:grand\s*total|(?:net|nett)\s*(?:total|amount)|"
    rf"total\s*(?:amount|due|payable|bayar|akhir)|amount\s*(?:due|payable)|"
    rf"jumlah\s*bayar|total)"
    rf"\s*(?:\({CURRENCY}\))?\s*[:=]?\s*(?:{CURRENCY}\s*)?"
    rf"(?P<amount>{VALUE})\s*(?:{CURRENCY})?\s*$",
    re.IGNORECASE,
)
ANY_VALUE = re.compile(VALUE)


@dataclass(frozen=True)
class TextRegion:
    text: str
    bbox: tuple[float, float, float, float]


def verify_total(predicted, regions):
    """Accept only a model total matching the unique OCR payable-total amount."""
    candidates = {}
    for region in regions:
        match = PAYABLE_LINE.fullmatch(region.text)
        if match:
            amount = number(match["amount"])
            if amount is not None:
                candidates[amount] = region

    status = "review"
    evidence = None
    if predicted is None:
        reason = "model_total_missing"
    elif not candidates:
        reason = "payable_label_missing"
    elif len(candidates) != 1:
        reason = "payable_amount_ambiguous"
    else:
        amount, region = next(iter(candidates.items()))
        evidence = {"amount": amount, "text": region.text, "bbox": region.bbox}
        if predicted == amount:
            status, reason = "verified", "payable_amount_matches"
        else:
            reason = "payable_amount_differs"
    return {"status": status, "reason": reason, "evidence": evidence}


def appears_anywhere(predicted, regions):
    """Weaker comparison policy that ignores the text around a matching number."""
    return predicted is not None and any(
        number(match[0]) == predicted
        for region in regions
        for match in ANY_VALUE.finditer(region.text)
    )
