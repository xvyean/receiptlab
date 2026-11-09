import json
import unittest

from receiptlab.data import annotation
from receiptlab.metrics import per_receipt, summarize
from receiptlab.ocr import extract_lines
from receiptlab.schema import checks, number, parse_prediction, to_csv


class ReceiptTests(unittest.TestCase):
    def test_amount_formats(self):
        self.assertEqual(number("Rp 25.500"), "25500")
        self.assertEqual(number("25.500,50"), "25500.5")
        self.assertEqual(number("1.250", amount=False), "1.25")
        self.assertIsNone(number("2x"))

    def test_predictions_validate_and_keep_decimal_values(self):
        parsed = parse_prediction('{"items":[],"subtotal":null,"total":"1.234"}')
        self.assertEqual(parsed["total"], "1.234")
        self.assertIsNone(parse_prediction('{"items":[],"subtotal":null,"total":null,"total":1}'))
        self.assertIsNone(parse_prediction('{"items":[],"subtotal":null,"total":true}'))
        self.assertIsNone(parse_prediction('{"items":[],"subtotal":null,"total":"1,000"}'))

    def test_cord_mapping_and_review(self):
        gt = {"gt_parse": {"menu": {"nm": "Tea", "cnt": "2", "unitprice": "5,000",
                                    "price": "9,500", "itemsubtotal": "10,000"},
                           "sub_total": {"subtotal_price": "10,000"},
                           "total": {"total_price": "10,500"}}}
        parsed = annotation(gt)
        self.assertEqual(parsed["items"][0]["line_total"], "10000")
        self.assertEqual(checks(parsed), [])
        parsed["items"][0]["line_total"] = "9000"
        self.assertEqual([issue["kind"] for issue in checks(parsed)],
                         ["line_amount_mismatch", "subtotal_mismatch"])
        self.assertEqual(parse_prediction(json.dumps(parsed)), parsed)

    def test_duplicate_rows_are_counted(self):
        item = {"name": "Tea", "quantity": "1", "unit_price": "5", "line_total": "5"}
        gold = {"items": [item, item], "subtotal": "10", "total": "10"}
        pred = {"items": [item], "subtotal": "5", "total": "5"}
        summary = summarize([per_receipt(gold, pred, 1)])
        self.assertAlmostEqual(summary["line_item_exact_f1"], 2 / 3)
        self.assertEqual(summary["total_exact_accuracy"], 0)

    def test_ocr_keeps_rows_and_total(self):
        receipt = extract_lines(["Tea 2 5,000 10,000", "SUBTOTAL 10,000",
                                 "SERVICE 2,000", "TOTAL 12,000", "CASH 20,000"])
        self.assertEqual(receipt["items"][0]["quantity"], "2")
        self.assertEqual(receipt["total"], "12000")
        self.assertEqual(receipt["subtotal"], "10000")

    def test_csv_does_not_evaluate_product_name(self):
        item = {"name": "=3+3", "quantity": "1", "unit_price": "5", "line_total": "5"}
        self.assertIn("'=3+3,", to_csv({"items": [item]}))


if __name__ == "__main__":
    unittest.main()
