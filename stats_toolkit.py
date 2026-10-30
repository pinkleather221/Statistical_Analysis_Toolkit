"""Public root-level interface for the statistics toolkit."""

from statistical_analysis_toolkit.stats_toolkit import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_PATH,
    QUARTILE_METHOD,
    StatisticsAnalyzer,
    compare_with_numpy,
    iqr_outliers,
    load_column,
    mean,
    median,
    mode,
    quartiles,
    std_dev,
    variance,
    main,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_OUTPUT_PATH",
    "QUARTILE_METHOD",
    "StatisticsAnalyzer",
    "compare_with_numpy",
    "iqr_outliers",
    "load_column",
    "mean",
    "median",
    "mode",
    "quartiles",
    "std_dev",
    "variance",
    "main",
]


if __name__ == "__main__":
    main()
