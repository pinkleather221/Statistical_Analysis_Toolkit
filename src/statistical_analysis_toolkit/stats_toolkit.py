"""First-principles statistics tools for the Wakala Plus deposit dataset."""

from __future__ import annotations

import csv
import io
import math
import unittest
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "agent_deposits.csv"
DEFAULT_OUTPUT_PATH = ROOT / "output" / "stats_report.txt"
QUARTILE_METHOD = "linear interpolation (NumPy method='linear')"


def _as_values(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("values must contain only finite numbers")
    return result


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean, calculated without statistics or NumPy."""
    data = _as_values(values)
    if not data:
        raise ValueError("mean requires at least one value")
    return sum(data) / len(data)


def median(values: Iterable[float]) -> float:
    """Return the middle value, or the mean of the two middle values."""
    data = sorted(_as_values(values))
    if not data:
        raise ValueError("median requires at least one value")
    middle = len(data) // 2
    if len(data) % 2:
        return data[middle]
    return (data[middle - 1] + data[middle]) / 2


def mode(values: Iterable[float]) -> list[float]:
    """Return all most-frequent values; return [] when there is no repeat."""
    data = _as_values(values)
    if not data:
        return []
    counts: dict[float, int] = {}
    for value in data:
        counts[value] = counts.get(value, 0) + 1
    highest_count = max(counts.values())
    if highest_count == 1:
        return []
    return sorted(value for value, count in counts.items() if count == highest_count)


def variance(values: Iterable[float], sample: bool) -> float:
    """Return sample (n-1) or population (n) variance using two passes."""
    data = _as_values(values)
    denominator = len(data) - 1 if sample else len(data)
    if denominator <= 0:
        size_requirement = "at least two" if sample else "at least one"
        raise ValueError(f"variance requires {size_requirement} value(s)")
    average = sum(data) / len(data)
    return sum((value - average) ** 2 for value in data) / denominator


def std_dev(values: Iterable[float], sample: bool) -> float:
    """Return the sample or population standard deviation."""
    return math.sqrt(variance(values, sample))


def _quantile_sorted(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )


def quartiles(values: Iterable[float]) -> tuple[float, float, float]:
    """Return Q1, median, Q3 using linear interpolation at (n-1)*p."""
    data = sorted(_as_values(values))
    if not data:
        raise ValueError("quartiles requires at least one value")
    return (
        _quantile_sorted(data, 0.25),
        _quantile_sorted(data, 0.5),
        _quantile_sorted(data, 0.75),
    )


def _iqr_outliers_sorted(
    sorted_values: Sequence[float], k: float
) -> dict[str, float | list[float]]:
    if not sorted_values:
        raise ValueError("outlier detection requires at least one value")
    if not math.isfinite(k) or k < 0:
        raise ValueError("k must be a finite, non-negative number")
    q1 = _quantile_sorted(sorted_values, 0.25)
    q3 = _quantile_sorted(sorted_values, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    outliers = [
        value
        for value in sorted_values
        if value < lower_fence or value > upper_fence
    ]
    return {
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outliers": outliers,
    }


def iqr_outliers(values: Iterable[float], k: float = 1.5) -> dict[str, float | list[float]]:
    """Return IQR, lower/upper fences, and values beyond either fence."""
    return _iqr_outliers_sorted(sorted(_as_values(values)), k)


def _read_column(
    path: str | Path, column: str
) -> tuple[list[float], list[tuple[int, str]], list[str], list[tuple[int, str, str]]]:
    values: list[float] = []
    rejected: list[tuple[int, str]] = []
    labels: list[str] = []
    rejection_details: list[tuple[int, str, str]] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"CSV does not contain column {column!r}")
        for row_number, row in enumerate(reader, start=2):
            raw_value = row.get(column, "")
            if raw_value is None or not raw_value.strip():
                reason = "missing value"
                rejected.append((row_number, raw_value or ""))
                rejection_details.append((row_number, raw_value or "", reason))
                continue
            try:
                value = float(raw_value.strip())
            except ValueError:
                reason = "not a valid number"
                rejected.append((row_number, raw_value))
                rejection_details.append((row_number, raw_value, reason))
                continue
            if not math.isfinite(value):
                reason = "not a finite number"
                rejected.append((row_number, raw_value))
                rejection_details.append((row_number, raw_value, reason))
                continue
            values.append(value)
            date = row.get("date", "")
            agent_id = row.get("agent_id", "")
            label = ", ".join(part for part in (date, agent_id) if part)
            labels.append(label)
    return values, rejected, labels, rejection_details


def load_column(path: str | Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Load finite numeric cells and rejected (CSV row number, raw value) pairs."""
    values, rejected, _, _ = _read_column(path, column)
    return values, rejected


class StatisticsAnalyzer:
    """Cache a dataset's sorted values and derived statistics for reuse."""

    def __init__(
        self,
        values: Iterable[float],
        name: str = "values",
        rejected: Sequence[tuple[int, str]] = (),
        rejection_details: Sequence[tuple[int, str, str]] = (),
        labels: Sequence[str] = (),
        rows_read: int | None = None,
    ) -> None:
        self.values = _as_values(values)
        if not self.values:
            raise ValueError("StatisticsAnalyzer requires at least one valid value")
        self.name = name
        self.rejected = list(rejected)
        self.rejection_details = list(rejection_details)
        self.labels = list(labels) if labels else [""] * len(self.values)
        if len(self.labels) != len(self.values):
            raise ValueError("labels must have the same length as values")
        self.rows_read = (
            rows_read if rows_read is not None else len(self.values) + len(self.rejected)
        )
        self._sorted_values = sorted(self.values)
        self._summary = self._calculate_summary()

    @classmethod
    def from_csv(cls, path: str | Path, column: str) -> StatisticsAnalyzer:
        values, rejected, labels, rejection_details = _read_column(path, column)
        return cls(
            values=values,
            name=column,
            rejected=rejected,
            rejection_details=rejection_details,
            labels=labels,
            rows_read=len(values) + len(rejected),
        )

    def _calculate_summary(
        self,
    ) -> dict[str, float | int | list[float] | None]:
        count = len(self._sorted_values)
        average = sum(self._sorted_values) / count
        median_value = _quantile_sorted(self._sorted_values, 0.5)
        counts: dict[float, int] = {}
        for value in self.values:
            counts[value] = counts.get(value, 0) + 1
        largest_count = max(counts.values())
        modes = (
            sorted(value for value, occurrences in counts.items() if occurrences == largest_count)
            if largest_count > 1
            else []
        )
        sample_variance = (
            sum((value - average) ** 2 for value in self.values) / (count - 1)
            if count > 1
            else None
        )
        q1 = _quantile_sorted(self._sorted_values, 0.25)
        q3 = _quantile_sorted(self._sorted_values, 0.75)
        outlier_info = _iqr_outliers_sorted(self._sorted_values, 1.5)
        return {
            "count": count,
            "mean": average,
            "median": median_value,
            "mode": modes,
            "variance": sample_variance,
            "std_dev": math.sqrt(sample_variance) if sample_variance is not None else None,
            "min": self._sorted_values[0],
            "max": self._sorted_values[-1],
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "lower_fence": float(outlier_info["lower_fence"]),
            "upper_fence": float(outlier_info["upper_fence"]),
            "outliers": list(outlier_info["outliers"]),
        }

    def summary(self) -> dict[str, float | int | list[float] | None]:
        """Return a shallow copy of the cached statistics summary."""
        return {
            key: value.copy() if isinstance(value, list) else value
            for key, value in self._summary.items()
        }

    def without_outliers(self) -> StatisticsAnalyzer:
        outliers = set(self._summary["outliers"])
        kept = [
            (value, label)
            for value, label in zip(self.values, self.labels)
            if value not in outliers
        ]
        if not kept:
            raise ValueError("outlier removal would leave no values to analyze")
        return StatisticsAnalyzer(
            values=[value for value, _ in kept],
            name=f"{self.name} (without outliers)",
            labels=[label for _, label in kept],
        )

    def report(
        self,
        comparison: dict[str, str] | None = None,
        test_results: str | None = None,
    ) -> str:
        return _format_report(
            self,
            comparison or compare_with_numpy(self),
            test_results,
        )


def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, str]:
    """Verify first-principles results; this is the only NumPy import."""
    import numpy as np

    summary = analyzer.summary()
    values = np.asarray(analyzer.values, dtype=float)
    q1, q2, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    unique_values, counts = np.unique(values, return_counts=True)
    largest_count = int(counts.max())
    numpy_modes = (
        unique_values[counts == largest_count].tolist() if largest_count > 1 else []
    )
    comparisons: list[tuple[str, float | None, float | None]] = [
        ("mean", float(summary["mean"]), float(np.mean(values))),
        ("median", float(summary["median"]), float(np.median(values))),
        (
            "variance",
            float(summary["variance"]) if summary["variance"] is not None else None,
            float(np.var(values, ddof=1)) if len(values) > 1 else None,
        ),
        (
            "std dev",
            float(summary["std_dev"]) if summary["std_dev"] is not None else None,
            float(np.std(values, ddof=1)) if len(values) > 1 else None,
        ),
        ("mode", 0.0, 0.0),
        ("Q1", float(summary["q1"]), float(q1)),
        ("Q2", float(summary["median"]), float(q2)),
        ("Q3", float(summary["q3"]), float(q3)),
        ("IQR", float(summary["iqr"]), float(q3 - q1)),
    ]
    results: dict[str, str] = {}
    for name, actual, expected in comparisons:
        if name == "mode":
            matches = [float(value) for value in summary["mode"]] == [
                float(value) for value in numpy_modes
            ]
        elif actual is None or expected is None:
            results[name] = "NOT AVAILABLE — sample requires at least two observations"
            continue
        else:
            matches = math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        if matches:
            results[name] = "MATCH"
        elif name == "variance":
            results[name] = (
                f"MISMATCH — ours={actual:.12g}, NumPy={expected:.12g}; "
                "sample variance requires ddof=1"
            )
        elif name == "std dev":
            results[name] = (
                f"MISMATCH — ours={actual:.12g}, NumPy={expected:.12g}; "
                "sample standard deviation requires ddof=1"
            )
        elif name in {"Q1", "Q2", "Q3", "IQR"}:
            results[name] = (
                f"MISMATCH — ours={actual:.12g}, NumPy={expected:.12g}; "
                f"quartile method is {QUARTILE_METHOD}"
            )
        else:
            results[name] = f"MISMATCH — ours={actual:.12g}, NumPy={expected:.12g}"
    return results


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def _format_report(
    analyzer: StatisticsAnalyzer,
    comparison: dict[str, str],
    test_results: str | None,
) -> str:
    cleaned = analyzer.without_outliers()
    all_summary = analyzer.summary()
    clean_summary = cleaned.summary()

    def paired(key: str) -> str:
        return f"{_format_number(all_summary[key]):>15}  {_format_number(clean_summary[key]):>18}"

    lines = [
        f"=== Statistics report: {analyzer.name} ===",
        f"Rows read         : {analyzer.rows_read}",
        f"Valid values      : {len(analyzer.values)}",
        f"Rejected values   : {len(analyzer.rejected)}",
    ]
    for row_number, raw_value, reason in analyzer.rejection_details:
        lines.append(f"  row {row_number}: {raw_value!r} ({reason})")
    lines.extend(
        [
            "",
            f"{'':20}{'all data':>15}  {'outliers removed':>18}",
            f"{'count':20}{paired('count')}",
            f"{'mean':20}{paired('mean')}",
            f"{'median':20}{paired('median')}",
            f"{'mode':20}{str(all_summary['mode'] or 'none'):>15}  "
            f"{str(clean_summary['mode'] or 'none'):>18}",
            f"{'variance (sample)':20}{paired('variance')}",
            f"{'std dev (sample)':20}{paired('std_dev')}",
            f"{'min / max':20}"
            f"{_format_number(all_summary['min'])} / {_format_number(all_summary['max']):>7}  "
            f"{_format_number(clean_summary['min'])} / {_format_number(clean_summary['max']):>7}",
            f"{'Q1 / Q3':20}"
            f"{_format_number(all_summary['q1'])} / {_format_number(all_summary['q3']):>7}  "
            f"{_format_number(clean_summary['q1'])} / {_format_number(clean_summary['q3']):>7}",
            f"{'IQR':20}{paired('iqr')}",
            "",
            f"Outlier fences      : lower {_format_number(all_summary['lower_fence'])}, "
            f"upper {_format_number(all_summary['upper_fence'])}",
        ]
    )
    outlier_labels: list[str] = []
    for value, label in zip(analyzer.values, analyzer.labels):
        if value in all_summary["outliers"]:
            suffix = f" ({label})" if label else ""
            outlier_labels.append(f"{_format_number(value)}{suffix}")
    lines.extend(
        [
            f"Outliers            : {', '.join(outlier_labels) if outlier_labels else 'none'}",
            "",
            "=== NumPy verification ===",
            f"variance/std dev use ddof=1; quartiles use {QUARTILE_METHOD}.",
        ]
    )
    for statistic, result in comparison.items():
        lines.append(f"{statistic:<10} {result}")
    lines.extend(
        [
            "",
            "Risk team notes:",
            "- The unusually high deposits may reflect legitimate large transactions, fraud, or a recording error; confirm them against transaction records.",
            "- The unusually low deposit may indicate a float shortage, a genuinely quiet day, or an input problem, so compare it with the agent's activity.",
            "- IQR flags are investigation prompts rather than proof of misconduct, and daily totals alone cannot identify the cause.",
            "",
        ]
    )
    lines.extend(
        [
            "=== Unit tests ===",
            test_results
            if test_results is not None
            else "Not run; use: python -m unittest -v test_stats_toolkit",
            "",
        ]
    )
    return "\n".join(lines)


def _run_unit_tests() -> tuple[str, bool]:
    result_stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(ROOT),
        pattern="test_stats_toolkit.py",
        top_level_dir=str(ROOT),
    )
    result = unittest.TextTestRunner(stream=result_stream, verbosity=0).run(suite)
    return result_stream.getvalue().strip(), result.wasSuccessful()


def main() -> None:
    analyzer = StatisticsAnalyzer.from_csv(DEFAULT_DATA_PATH, "daily_deposits_kes")
    test_results, tests_passed = _run_unit_tests()
    report_text = analyzer.report(test_results=test_results)
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nSaved report to {DEFAULT_OUTPUT_PATH}")
    if not tests_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
