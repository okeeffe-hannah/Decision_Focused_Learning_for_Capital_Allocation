"""Download and clean the Federal Reserve Excess Bond Premium dataset."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


EBP_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "ebp.csv"


def _download_ebp_bytes(url: str) -> bytes:
    """Download the EBP CSV bytes with a browser-like user agent."""
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Federal Reserve returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not download EBP dataset: {exc.reason}") from exc


def _read_ebp_csv(url: str) -> pd.DataFrame:
    """Read the EBP CSV while tolerating metadata rows before the header."""
    csv_bytes = _download_ebp_bytes(url)
    csv_buffer = BytesIO(csv_bytes)

    try:
        raw = pd.read_csv(csv_buffer)
    except pd.errors.ParserError:
        csv_buffer.seek(0)
        raw = pd.read_csv(csv_buffer, header=None)

    if "date" in [str(column).strip().lower() for column in raw.columns]:
        return raw

    csv_buffer.seek(0)
    raw_no_header = pd.read_csv(csv_buffer, header=None)
    header_row = raw_no_header.apply(
        lambda row: row.astype(str).str.strip().str.lower().eq("date").any(),
        axis=1,
    )
    if not header_row.any():
        raise ValueError("Could not identify the EBP CSV header row")

    header_index = header_row.idxmax()
    columns = raw_no_header.iloc[header_index].astype(str).str.strip().tolist()
    cleaned = raw_no_header.iloc[header_index + 1 :].copy()
    cleaned.columns = columns
    return cleaned


def _clean_ebp_dataframe(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Return a cleaned EBP dataframe with DATE and EBP columns."""
    raw = raw.copy()
    raw.columns = [str(column).strip().lower() for column in raw.columns]

    if "date" not in raw.columns:
        raise ValueError(f"{source_label} EBP data does not contain a date column")
    if "ebp" not in raw.columns:
        raise ValueError(f"{source_label} EBP data does not contain an ebp column")

    cleaned = raw[["date", "ebp"]].rename(columns={"date": "DATE", "ebp": "EBP"})
    cleaned["DATE"] = pd.to_datetime(cleaned["DATE"], errors="coerce")
    cleaned["EBP"] = pd.to_numeric(cleaned["EBP"], errors="coerce")
    cleaned = cleaned.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)

    if cleaned.empty:
        raise ValueError(f"No valid EBP observations were found in {source_label} data")

    return cleaned


def load_ebp(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load, clean, and save the Excess Bond Premium series.

    The function reuses an existing local CSV by default. Set
    `force_download=True` to refresh it from the Federal Reserve source.

    Parameters
    ----------
    output_path:
        Destination CSV path. Defaults to `data/raw/ebp.csv`.
    force_download:
        Redownload the dataset even when `output_path` already exists.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with `DATE` and `EBP` columns.

    Raises
    ------
    ValueError
        If the downloaded file does not contain recognizable date or EBP columns.
    RuntimeError
        If the dataset cannot be downloaded.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force_download:
        raw = pd.read_csv(output_path)
        return _clean_ebp_dataframe(raw, "cached")

    raw = _read_ebp_csv(EBP_URL)
    cleaned = _clean_ebp_dataframe(raw, "downloaded")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for downloading EBP data."""
    parser = argparse.ArgumentParser(
        description="Download and clean the Federal Reserve EBP dataset."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination CSV path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the EBP CSV even if the output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Run EBP download and cleaning from the command line."""
    args = parse_args()
    cleaned = load_ebp(args.output_path, force_download=args.force_download)
    print(f"Saved {len(cleaned)} EBP rows to {args.output_path}")


if __name__ == "__main__":
    main()
