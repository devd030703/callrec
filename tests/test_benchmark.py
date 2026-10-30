import unittest

from benchmarks.score import normalise, score


class BenchmarkTests(unittest.TestCase):
    def test_preserves_scripts_and_accents(self):
        self.assertNotEqual(normalise("français"), normalise("francais"))
        self.assertNotEqual(normalise("मैं चाय"), normalise("main chai"))

    def test_promotion_requires_improvement_and_closing(self):
        rows = [
            {
                "id": "a",
                "language": "Hindi-English",
                "reference": "नमस्ते final sentence",
                "baseline": "नमस्ते",
                "candidate": "नमस्ते final sentence",
                "required_phrases": ["final sentence"],
            }
        ]
        self.assertTrue(score(rows)["promotion_gate_passed"])
        rows[0]["candidate"] = "नमस्ते"
        self.assertFalse(score(rows)["promotion_gate_passed"])

    def test_no_regression_can_be_hidden_by_another_language(self):
        rows = [
            {
                "id": "a",
                "language": "fr",
                "reference": "bonjour ami",
                "baseline": "bonjour ami",
                "candidate": "bonjour ennemi",
            },
            {
                "id": "b",
                "language": "hi",
                "reference": "नमस्ते दोस्त",
                "baseline": "ग़लत",
                "candidate": "नमस्ते दोस्त",
            },
        ]
        result = score(rows)
        self.assertFalse(result["promotion_gate_passed"])
        self.assertIn("wer_regression_over_one_point", [f["reason"] for f in result["failures"]])

    def test_duplicate_ids_are_rejected(self):
        row = {"id": "a", "language": "en", "reference": "hi", "baseline": "hi", "candidate": "hi"}
        with self.assertRaises(ValueError):
            score([row, row])
