import tempfile
import unittest
from pathlib import Path

from train_traditional_classification import RepresentationStats, load_split_file, split_train_val


class TraditionalClassificationHelperTest(unittest.TestCase):
    def test_split_train_val_is_deterministic_and_disjoint(self):
        train_a, val_a = split_train_val(list(range(20)), val_fraction=0.2, seed=42)
        train_b, val_b = split_train_val(list(range(20)), val_fraction=0.2, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(val_a), 4)
        self.assertFalse(set(train_a) & set(val_a))
        self.assertEqual(sorted(train_a + val_a), list(range(20)))

    def test_load_split_file_reads_int_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split.json"
            path.write_text('{"train": [1, "2"], "test": [3, "4"]}', encoding="utf-8")
            train, test = load_split_file(path)
        self.assertEqual(train, [1, 2])
        self.assertEqual(test, [3, 4])

    def test_representation_stats_summary(self):
        stats = RepresentationStats()
        stats.update_batch(
            [
                {
                    "num_events": 10,
                    "build_seconds": 0.1,
                    "shape": [2, 3, 4],
                    "nonzero_ratio": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "mean": 0.25,
                    "std": 0.1,
                },
                {
                    "num_events": 30,
                    "build_seconds": 0.3,
                    "shape": [2, 3, 4],
                    "nonzero_ratio": 0.25,
                    "min": 0.0,
                    "max": 2.0,
                    "mean": 0.5,
                    "std": 0.2,
                },
            ]
        )
        summary = stats.to_dict()
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["mean_events"], 20.0)
        self.assertEqual(summary["shape_counts"], {"2x3x4": 2})
        self.assertAlmostEqual(summary["mean_nonzero_ratio"], 0.375)


if __name__ == "__main__":
    unittest.main()
