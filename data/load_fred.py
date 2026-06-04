"""Download FRED time series data to the local raw data directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _download_from_fred_api(series_id: str, api_key: str) -> pd.DataFrame:
    """Download a FRED series through the official API."""
    query = urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }
    )
    url = f"{FRED_API_URL}?{query}"

    try:
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"FRED API returned HTTP {exc.code} for {series_id}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not download FRED series {series_id}: {exc.reason}") from exc

    if "observations" not in payload:
        message = payload.get("error_message", "missing observations payload")
        raise RuntimeError(f"FRED API response for {series_id} was invalid: {message}")

    observations = payload["observations"]
    return pd.DataFrame(
        {
            "DATE": [observation["date"] for observation in observations],
            series_id: [observation["value"] for observation in observations],
        }
    )


def _download_from_public_csv(series_id: str) -> pd.DataFrame:
    """Download a FRED series from the public graph CSV endpoint."""
    url = f"{FRED_CSV_URL}?{urlencode({'id': series_id})}"

    try:
        df = pd.read_csv(url)
    except HTTPError as exc:
        raise RuntimeError(f"FRED returned HTTP {exc.code} for series {series_id}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not download FRED series {series_id}: {exc.reason}") from exc

    return df


def _normalize_fred_dataframe(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Return a clean two-column dataframe with DATE and the FRED series ID."""
    rename_map = {
        column: "DATE"
        for column in df.columns
        if str(column).strip().lower() in {"date", "observation_date"}
    }
    df = df.rename(columns=rename_map)

    if "DATE" not in df.columns:
        raise ValueError(f"Downloaded FRED data for {series_id} does not contain a date column")

    value_columns = [column for column in df.columns if column != "DATE"]
    if series_id in value_columns:
        value_column = series_id
    elif len(value_columns) == 1:
        value_column = value_columns[0]
    else:
        raise ValueError(
            f"Downloaded FRED data for {series_id} must contain exactly one value column"
        )

    cleaned = df[["DATE", value_column]].rename(columns={value_column: series_id})
    cleaned["DATE"] = pd.to_datetime(cleaned["DATE"], errors="coerce")
    cleaned[series_id] = pd.to_numeric(cleaned[series_id].replace(".", pd.NA), errors="coerce")
    cleaned = cleaned.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)

    if cleaned.empty:
        raise ValueError(f"No valid observations found for FRED series {series_id}")

    return cleaned


def download_fred_series(
    series_id: str,
    output_dir: Path = DEFAULT_RAW_DIR,
    api_key: str | None = None,
    force_download: bool = False,
) -> Path:
    """Download a FRED series as a normalized CSV and return the saved file path.

    If `api_key` is omitted, the function uses the `FRED_API_KEY` environment
    variable when present. Without an API key, it falls back to FRED's public CSV
    endpoint. Existing local CSV files are reused by default; set
    `force_download=True` to refresh a series from FRED.
    """
    normalized_series_id = series_id.strip().upper()
    if not normalized_series_id:
        raise ValueError("series_id must not be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{normalized_series_id}.csv"
    if output_path.exists() and not force_download:
        cached = pd.read_csv(output_path)
        _normalize_fred_dataframe(cached, normalized_series_id)
        return output_path

    api_key = api_key or os.getenv("FRED_API_KEY")
    if api_key:
        raw = _download_from_fred_api(normalized_series_id, api_key)
    else:
        raw = _download_from_public_csv(normalized_series_id)

    cleaned = _normalize_fred_dataframe(raw, normalized_series_id)
    cleaned.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for downloading a FRED series."""
    parser = argparse.ArgumentParser(
        description="Download a FRED series and save it under data/raw as CSV."
    )
    parser.add_argument("series_id", help="FRED series ID, for example GDP or CPIAUCSL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory to save the CSV file. Defaults to {DEFAULT_RAW_DIR}.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="FRED API key. Defaults to the FRED_API_KEY environment variable.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the series even if the local raw CSV already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Download a FRED series from the command line."""
    args = parse_args()
    output_path = download_fred_series(
        args.series_id,
        args.output_dir,
        args.api_key,
        force_download=args.force_download,
    )
    print(f"Saved FRED series to {output_path}")


if __name__ == "__main__":
    main()
