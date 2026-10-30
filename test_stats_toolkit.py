import csv
import tempfile
import unittest
from pathlib import Path

from stats_toolkit import (
    StatisticsAnalyzer,
    iqr_outliers,
    load_column,
    mean,
    median,
    mode,
    quartiles,
    std_dev,
    variance,
)


class TestStatisticsFunctions(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([2, 4, 6]), 4)

    def test_mean_empty_raises(self):
        with self.assertRaises(ValueError):
            mean([])

    def test_median_odd_length(self):
        self.assertEqual(median([9, 1, 5]), 5)

    def test_median_even_length(self):
        self.assertEqual(median([8, 2, 4, 6]), 5)

    def test_mode_returns_all_tied_modes(self):
        self.assertEqual(mode([1, 1, 2, 2, 3]), [1.0, 2.0])

    def test_mode_is_empty_when_values_do_not_repeat(self):
        self.assertEqual(mode([1, 2, 3]), [])
        self.assertEqual(mode([]), [])

    def test_sample_and_population_variance(self):
        self.assertEqual(variance([2, 4, 6], sample=True), 4)
        self.assertAlmostEqual(variance([2, 4, 6], sample=False), 8 / 3)

    def test_sample_and_population_standard_deviation(self):
        self.assertEqual(std_dev([2, 4, 6], sample=True), 2)
        self.assertAlmostEqual(std_dev([2, 4, 6], sample=False), (8 / 3) ** 0.5)

    def test_sample_variance_needs_two_values(self):
        with self.assertRaises(ValueError):
            variance([1], sample=True)

    def test_linear_interpolation_quartiles(self):
        self.assertEqual(quartiles([1, 2, 3, 4, 5]), (2, 3, 4))

    def test_iqr_outliers_include_fences_and_extremes(self):
        result = iqr_outliers([1, 2, 3, 4, 5, 100])
        self.assertEqual(result["iqr"], 2.5)
        self.assertAlmostEqual(result["lower_fence"], -1.5)
        self.assertAlmostEqual(result["upper_fence"], 8.5)
        self.assertEqual(result["outliers"], [100.0])

    def test_load_column_reports_invalid_values_and_row_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "sample.csv"
            csv_path.write_text(
                'amount\n10\n""\nnope\n20\nNaN\n',
                encoding="utf-8",
            )
            values, rejected = load_column(csv_path, "amount")
        self.assertEqual(values, [10.0, 20.0])
        self.assertEqual(
            rejected,
            [(3, ""), (4, "nope"), (6, "NaN")],
        )

    def test_from_csv_reports_rejection_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "sample.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["date", "agent_id", "amount"])
                writer.writerow(["2026-01-01", "AG-1", "12"])
                writer.writerow(["2026-01-02", "AG-2", ""])
                writer.writerow(["2026-01-03", "AG-3", "oops"])
            analyzer = StatisticsAnalyzer.from_csv(csv_path, "amount")
        report = analyzer.report()
        self.assertIn("row 3: '' (missing value)", report)
        self.assertIn("row 4: 'oops' (not a valid number)", report)

    def test_without_outliers_returns_a_new_analyzer(self):
        analyzer = StatisticsAnalyzer([1, 2, 3, 4, 100])
        cleaned = analyzer.without_outliers()
        self.assertIsNot(analyzer, cleaned)
        self.assertEqual(cleaned.values, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(analyzer.values, [1.0, 2.0, 3.0, 4.0, 100.0])


if __name__ == "__main__":
    unittest.main()
