"""Preprocess raw FRED CSV files into a merged monthly dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merged_monthly.csv"
DEFAULT_SERIES_CONFIG_PATH = PROJECT_ROOT / "configs" / "macrofinancial_series.csv"
VALID_HIGH_FREQUENCY_AGGREGATIONS = {"mean", "last", "first", "median"}
VALID_MONTHLY_FILL_METHODS = {"ffill"}


def _read_fred_csv(csv_path: Path) -> pd.DataFrame:
    """Read a raw FRED CSV file and return a two-column DATE/value dataframe."""
    try:
        df = pd.read_csv(csv_path)
    except OSError as exc:
        raise OSError(f"Could not read {csv_path}: {exc}") from exc

    date_columns = [
        column
        for column in df.columns
        if str(column).strip().lower() in {"date", "observation_date"}
    ]
    if len(date_columns) != 1:
        raise ValueError(f"{csv_path} must contain a DATE or observation_date column")

    df = df.rename(columns={date_columns[0]: "DATE"})
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    value_columns = [column for column in df.columns if column != "DATE"]
    if len(value_columns) != 1:
        raise ValueError(
            f"{csv_path} must contain exactly one value column besides DATE; "
            f"found {len(value_columns)}"
        )

    value_column = value_columns[0]
    if df["DATE"].isna().any():
        raise ValueError(f"{csv_path} contains invalid or missing DATE values")

    df[value_column] = pd.to_numeric(df[value_column].replace(".", pd.NA), errors="coerce")
    return df[["DATE", value_column]].sort_values("DATE")


def _to_month_end(df: pd.DataFrame, high_frequency_agg: str) -> pd.DataFrame:
    """Move observations to month-end timestamps and aggregate duplicates."""
    value_column = [column for column in df.columns if column != "DATE"][0]
    monthly = df.copy()
    monthly["DATE"] = monthly["DATE"] + pd.offsets.MonthEnd(0)

    if monthly["DATE"].duplicated().any():
        monthly = (
            monthly.groupby("DATE", as_index=False)[value_column]
            .agg(high_frequency_agg)
            .sort_values("DATE")
        )

    return monthly


def _load_monthly_fill_rules(series_config_path: str | Path | None) -> dict[str, str]:
    """Load per-series monthly fill rules from an optional CSV config."""
    if series_config_path is None:
        return {}

    series_config_path = Path(series_config_path)
    if not series_config_path.exists():
        raise FileNotFoundError(f"Series config does not exist: {series_config_path}")

    with series_config_path.open(newline="") as config_file:
        reader = csv.DictReader(config_file)
        required_columns = {"series_id", "monthly_fill"}
        if not required_columns.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"{series_config_path} must contain columns: {sorted(required_columns)}"
            )

        fill_rules = {}
        for row in reader:
            series_id = row["series_id"].strip()
            monthly_fill = row.get("monthly_fill", "").strip().lower()
            if not series_id or not monthly_fill:
                continue
            if monthly_fill not in VALID_MONTHLY_FILL_METHODS:
                raise ValueError(
                    f"Invalid monthly_fill '{monthly_fill}' for {series_id}; "
                    f"expected one of {sorted(VALID_MONTHLY_FILL_METHODS)}"
                )
            fill_rules[series_id.upper()] = monthly_fill

    return fill_rules


def _apply_monthly_fill_rules(
    merged: pd.DataFrame,
    fill_rules: dict[str, str],
) -> pd.DataFrame:
    """Apply monthly fill rules to aligned series after merging."""
    if not fill_rules:
        return merged

    merged = merged.sort_values("DATE").reset_index(drop=True)
    for series_id, fill_method in fill_rules.items():
        matching_columns = [
            column for column in merged.columns if column.upper() == series_id
        ]
        if not matching_columns:
            continue

        column = matching_columns[0]
        first_valid_date = merged.loc[merged[column].notna(), "DATE"].min()
        if pd.isna(first_valid_date):
            continue

        fill_mask = merged["DATE"] >= first_valid_date
        if fill_method == "ffill":
            merged.loc[fill_mask, column] = merged.loc[fill_mask, column].ffill()

    return merged


def preprocess_fred_series(
    input_dir: str | Path,
    output_path: str | Path,
    high_frequency_agg: str = "mean",
    series_config_path: str | Path | None = DEFAULT_SERIES_CONFIG_PATH,
) -> pd.DataFrame:
    """Merge raw FRED CSV files into a sorted monthly dataframe.

    Parameters
    ----------
    input_dir:
        Directory containing one or more raw FRED CSV files. Each file must have a
        DATE column and exactly one value column named with the FRED series ID.
    output_path:
        CSV path where the merged monthly dataframe will be written.
    high_frequency_agg:
        Aggregation used when a series has multiple observations in the same
        month after month-end alignment. Defaults to "mean".
    series_config_path:
        Optional CSV path with `series_id` and `monthly_fill` columns. A
        `monthly_fill` value of "ffill" forward-fills that series after merging,
        which is useful for lower-frequency inputs such as quarterly variables.

    Returns
    -------
    pandas.DataFrame
        The merged dataframe with DATE plus one column per FRED series.

    Raises
    ------
    FileNotFoundError
        If the input directory does not exist or contains no CSV files.
    ValueError
        If a CSV does not match the expected FRED structure.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    high_frequency_agg = high_frequency_agg.lower()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if high_frequency_agg not in VALID_HIGH_FREQUENCY_AGGREGATIONS:
        raise ValueError(
            "high_frequency_agg must be one of "
            f"{sorted(VALID_HIGH_FREQUENCY_AGGREGATIONS)}"
        )

    fill_rules = _load_monthly_fill_rules(series_config_path)

    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    monthly_frames = [
        _to_month_end(_read_fred_csv(csv_path), high_frequency_agg)
        for csv_path in csv_paths
    ]
    merged = monthly_frames[0]
    for frame in monthly_frames[1:]:
        merged = merged.merge(frame, on="DATE", how="outer")

    merged = merged.sort_values("DATE").reset_index(drop=True)
    merged = _apply_monthly_fill_rules(merged, fill_rules)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for preprocessing raw FRED CSV files."""
    parser = argparse.ArgumentParser(
        description="Merge raw FRED CSV files into a monthly dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing raw FRED CSV files. Defaults to {DEFAULT_INPUT_DIR}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the merged CSV. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--high-frequency-agg",
        choices=sorted(VALID_HIGH_FREQUENCY_AGGREGATIONS),
        default="mean",
        help="How to aggregate multiple observations in the same month. Defaults to mean.",
    )
    parser.add_argument(
        "--series-config-path",
        type=Path,
        default=DEFAULT_SERIES_CONFIG_PATH,
        help=(
            "Optional CSV with per-series monthly fill rules. "
            f"Defaults to {DEFAULT_SERIES_CONFIG_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run FRED preprocessing from the command line."""
    args = parse_args()
    merged = preprocess_fred_series(
        args.input_dir,
        args.output_path,
        high_frequency_agg=args.high_frequency_agg,
        series_config_path=args.series_config_path,
    )
    print(f"Saved {len(merged)} monthly rows to {args.output_path}")


if __name__ == "__main__":
    main()
