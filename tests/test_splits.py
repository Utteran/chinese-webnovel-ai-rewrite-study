"""Regression tests for the experimental split that defines each reported score."""

import json
import tempfile
import unittest
from pathlib import Path

from train.prepare_v2_splits import load_or_make_split, make_book_split


class SplitTests(unittest.TestCase):
    def test_input_order_does_not_change_generated_split(self):
        books = [f"book-{i}" for i in range(21)]
        self.assertEqual(make_book_split(books), make_book_split(reversed(books)))
        self.assertEqual(set(make_book_split(books).values()), {"train", "val", "test"})

    def test_existing_published_split_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            published = {"c": "train", "a": "val", "b": "test"}
            path.write_text(json.dumps({"book_split": published, "n_books": 3}),
                            encoding="utf-8")
            selected, created = load_or_make_split(["a", "b", "c"], manifest=path)
            self.assertFalse(created)
            self.assertEqual(selected, published)
            self.assertEqual(selected, json.loads(path.read_text(encoding="utf-8"))["book_split"])

    def test_book_set_change_requires_new_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps({"book_split": {
                "a": "train", "b": "val", "c": "test"
            }}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_or_make_split(["a", "b", "different"], manifest=path)


if __name__ == "__main__":
    unittest.main()
