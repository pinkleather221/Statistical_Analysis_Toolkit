# Wakala Plus Statistics Toolkit

This project analyzes the daily M-Pesa deposit totals in `data/agent_deposits.csv`.
All statistical calculations are implemented in plain Python in `stats_toolkit.py`.
NumPy is imported only by `compare_with_numpy()` as an independent verification
step; the data is not cleaned, summarized, or calculated with NumPy or pandas.

## Run the toolkit

Python 3.14 or newer is required (matching `pyproject.toml`). Install the verification dependency and run the
report generator from the repository root:

```powershell
python -m pip install -r requirements.txt
python stats_toolkit.py
```

The program runs the unit tests, prints the report (including their result), and
saves the same content to `output/stats_report.txt`. It exits with an error if
the tests fail.

## Run the tests

The tests use only hand-checkable examples and Python's built-in `unittest`:

```powershell
python -m unittest -v test_stats_toolkit
```

## Approach and assumptions

- `mean`, `median`, `mode`, `variance`, `std_dev`, quartiles, and IQR outlier
  fences are calculated directly with Python. Mean/median/variance require
  non-empty inputs; sample variance and standard deviation require at least two
  values. `mode` returns every tied most-frequent value, but returns an empty
  list when every value appears once (there is no useful mode to report).
- The report uses **sample variance and standard deviation (`n - 1`)** because
  these 40 dates are observations used to assess the broader behavior of agent
  deposits, rather than a complete population of every possible operating day.
  NumPy verification uses `ddof=1` to match.
- Quartiles use **linear interpolation at position `(n - 1) * p`**, also known
  as NumPy's `method="linear"` convention. This handles odd and even sample
  sizes consistently, and the verification uses the exact same method.
- CSV row numbers include the header as row 1. Blank cells, non-numeric text,
  and non-finite numbers such as `NaN` or `inf` are rejected, with each raw
  value and rejection reason recorded in the report. Values are not guessed or
  silently corrected; finite numeric values, including zero, are retained.
- Outliers are strictly below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR`.
  Quartile fences are flags for human investigation, not evidence of fraud.
- `StatisticsAnalyzer` computes and caches sorted values and summary measures
  when it is created. `without_outliers()` creates a separate analyzer and
  leaves the original data unchanged.

## Project files

- `stats_toolkit.py` — root-level public interface and report-generation entry
  point. The implementation is in `src/statistical_analysis_toolkit/`.
- `test_stats_toolkit.py` — unit tests for statistics, CSV rejection, and
  outlier behavior.
- `data/agent_deposits.csv` — supplied agent daily totals.
- `output/stats_report.txt` — generated report.
- `requirements.txt` — NumPy, used only for verification.
