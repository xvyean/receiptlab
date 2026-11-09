import math
import unittest

from receiptlab.ocr import grouped_regions
from receiptlab.review_eval import summarize_review
from receiptlab.verification import TextRegion, appears_anywhere, verify_total


def region(text):
    return TextRegion(text, (0.1, 0.2, 0.8, 0.3))


class VerificationTests(unittest.TestCase):
    def test_uses_payable_label_and_supports_currency(self):
        lines = [region("SUBTOTAL 12.00"), region("TOTAL GST 1.20"),
                 region("GRAND TOTAL (RM): 13.20"), region("CASH 20.00")]
        result = verify_total("13.2", lines)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["evidence"]["bbox"], (0.1, 0.2, 0.8, 0.3))
        self.assertTrue(appears_anywhere("12", lines))
        self.assertEqual(verify_total("12", lines)["reason"], "payable_amount_differs")

    def test_rejects_ambiguous_or_unlabeled_values(self):
        self.assertEqual(verify_total("5", [region("TAX 5"), region("ITEM TOTAL 5")])["reason"],
                         "payable_label_missing")
        self.assertEqual(verify_total("5", [region("TOTAL 5"), region("AMOUNT DUE 8")])["reason"],
                         "payable_amount_ambiguous")
        self.assertEqual(verify_total(None, [region("TOTAL 5")])["reason"],
                         "model_total_missing")

    def test_same_total_printed_twice_is_unique(self):
        result = verify_total("49000", [region("TOTAL Rp49.000"), region("TOTAL BAYAR 49,000")])
        self.assertEqual(result["status"], "verified")

    def test_bbox_covers_both_label_and_value(self):
        det = [([[8, 20], [34, 20], [34, 32], [8, 32]], "TOTAL", 1),
               ([[70, 21], [92, 21], [92, 32], [70, 32]], "5.00", 1)]
        lines = grouped_regions(det, 100, 100)
        self.assertEqual(lines, [TextRegion("TOTAL 5.00", (0.08, 0.2, 0.92, 0.32))])

    def test_skew_alignment_groups_label_with_its_own_amount(self):
        angle = math.radians(9)
        rotate = lambda x, y: [x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)]
        def quad(x, y, text):
            return ([rotate(x, y), rotate(x + 22, y), rotate(x + 22, y + 4), rotate(x, y + 4)], text, 1)
        detections = [quad(12, 15, "TOTAL"), quad(82, 15, "17000"),
                      quad(12, 26, "CASH"), quad(82, 26, "20000")]
        corrected = grouped_regions(detections, 130, 70, deskew=True)
        self.assertEqual([row.text for row in corrected], ["TOTAL 17000", "CASH 20000"])
        self.assertEqual(verify_total("17000", corrected)["status"], "verified")
        self.assertEqual(verify_total("20000", corrected)["reason"], "payable_amount_differs")

    def test_selective_scoring_excludes_unlabeled_examples(self):
        def record(gold, predicted, agree):
            return {"gold": gold, "prediction": predicted, "ocr_seconds": 0.3, "reason": "test",
                    "decisions": {"model_has_total": predicted is not None,
                                  "amount_anywhere": predicted is not None,
                                  "labeled_total_agreement": agree,
                                  "deskewed_total_agreement": agree}}
        result = summarize_review([record("8", "8", True), record("3", "8", False),
                                   record(None, "8", True), record("4", None, False)])
        self.assertEqual(result["eligible"], 3)
        policy = result["policies"]["labeled_total_agreement"]
        self.assertEqual(policy["coverage"], 1 / 3)
        self.assertEqual(policy["precision"], 1)
        self.assertEqual(policy["wrong_routed_to_review"], 2)


if __name__ == "__main__":
    unittest.main()
