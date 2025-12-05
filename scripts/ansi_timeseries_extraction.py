#!/usr/bin/env python3
"""Extract ANSI station time-series from MintPy LOS products.

The script samples MintPy geocoded LOS time-series for a fixed list of ANSI
stations, then exports per-station absolute and ST01-referenced displacement
CSV files as well as paired plots for both ascending and descending stacks.
Outputs follow the naming pattern ``{station}_{orbit}_...`` directly inside the
selected orbit directories.

Example
-------
python ansi_timeseries_extraction.py \\
    --ascending /path/to/ascending/geo_timeseries_ERA5_demErr.h5 \\
    --descending /path/to/descending/geo_timeseries_ERA5_demErr.h5 \\
    --ascending-output-dir /desired/path/ANSI32TS_ASC \\
    --descending-output-dir /desired/path/ANSI32TS_DSC
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class Station:
    name: str
    latitude: float
    longitude: float
    elevation: float


STATIONS: Tuple[Station, ...] = (
    Station("ST01", 37.736816, -6.616807, 344.0),
    Station("ST02", 37.736208, -6.619000, 349.5),
    Station("ST03", 37.737580, -6.613800, 348.7),
    Station("ST04", 37.736901, -6.614511, 347.7),
    Station("ST05", 37.736503, -6.615201, 347.4),
    Station("ST06", 37.736204, -6.615728, 350.1),
    Station("ST07", 37.735683, -6.616847, 365.8),
    Station("ST08", 37.737564, -6.612969, 363.2),
    Station("ST09", 37.737800, -6.611439, 380.1),
    Station("ST10", 37.737300, -6.612601, 379.2),
    Station("ST11", 37.736801, -6.613400, 379.9),
    Station("ST12", 37.736298, -6.614100, 378.6),
    Station("ST13", 37.735743, -6.615107, 378.2),
    Station("ST14", 37.735191, -6.616700, 377.2),
    Station("ST15", 37.737625, -6.610901, 377.3),
    Station("ST16", 37.737266, -6.611702, 387.5),
    Station("ST17", 37.736902, -6.612442, 384.1),
    Station("ST18", 37.736537, -6.613000, 385.8),
    Station("ST19", 37.736061, -6.613600, 386.0),
    Station("ST20", 37.735716, -6.614100, 385.7),
    Station("ST21", 37.735400, -6.614899, 386.1),
    Station("ST22", 37.735100, -6.615601, 385.5),
    Station("ST23", 37.734934, -6.616300, 385.5),
    Station("ST24", 37.734800, -6.617400, 386.7),
    Station("ST25", 37.735763, -6.618135, 354.7),
    Station("ST26", 37.735450, -6.619483, 355.1),
    Station("ST27", 37.735152, -6.620666, 353.9),
    Station("ST28", 37.734792, -6.618437, 386.4),
    Station("ST29", 37.734526, -6.619597, 388.6),
    Station("ST30", 37.734164, -6.621247, 384.1),
    Station("ST31", 37.736646, -6.615692, 400.0),
    Station("ST32", 37.737107, -6.614778, 400.0),
)


def _attrs_to_float(attrs: h5py.AttributeManager, key: str) -> float:
    """Return attribute value as float regardless of storage type."""
    value = attrs[key]
    if isinstance(value, (bytes, str)):
        return float(value)
    return float(np.array(value).squeeze())


def _latlon_to_rowcol(
    attrs: h5py.AttributeManager, latitude: float, longitude: float
) -> Tuple[int, int]:
    """Convert latitude/longitude to row/column indices of the geocoded grid."""
    x_first = _attrs_to_float(attrs, "X_FIRST")
    x_step = _attrs_to_float(attrs, "X_STEP")
    y_first = _attrs_to_float(attrs, "Y_FIRST")
    y_step = _attrs_to_float(attrs, "Y_STEP")
    width = int(_attrs_to_float(attrs, "WIDTH"))
    length = int(_attrs_to_float(attrs, "LENGTH"))

    col_float = (longitude - x_first) / x_step
    row_float = (latitude - y_first) / y_step

    col = int(np.rint(col_float))
    row = int(np.rint(row_float))

    if not (0 <= col < width and 0 <= row < length):
        raise ValueError(
            f"Coordinate ({latitude:.6f}, {longitude:.6f}) outside raster bounds: "
            f"row={row_float:.2f}, col={col_float:.2f}"
        )

    return row, col


def _read_timeseries(
    h5_path: Path, stations: Iterable[Station]
) -> pd.DataFrame:
    """Read station time-series from a MintPy geocoded HDF5 file."""
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as hf:
        ts_ds = hf["timeseries"]
        raw_dates = hf["date"][:]
        dates = pd.to_datetime(
            [d.decode("utf-8") for d in raw_dates], format="%Y%m%d"
        )

        attrs = hf.attrs
        no_data_value = None
        if "NO_DATA_VALUE" in attrs:
            try:
                no_data_value = _attrs_to_float(attrs, "NO_DATA_VALUE")
            except Exception:
                no_data_value = None

        series: Dict[str, np.ndarray] = {}

        for station in stations:
            row, col = _latlon_to_rowcol(attrs, station.latitude, station.longitude)

            data = ts_ds[:, row, col].astype(np.float64)
            if no_data_value is not None:
                data = np.where(data == no_data_value, np.nan, data)
            series[station.name] = data

    frame = pd.DataFrame(series, index=dates).sort_index()
    frame.index.name = "date"
    return frame


def _relative_to_reference(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Return a DataFrame of station displacements relative to the reference."""
    if reference not in frame.columns:
        raise KeyError(f"{reference} not found in columns {frame.columns.tolist()}")
    rel = frame.subtract(frame[reference], axis=0)
    return rel


def _filter_time_range(
    frame: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Subset a DataFrame by inclusive datetime bounds."""
    if start is None and end is None:
        return frame

    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame.index >= start
    if end is not None:
        mask &= frame.index <= end

    filtered = frame.loc[mask]
    if filtered.empty:
        raise ValueError(
            "Requested time range produced no data: "
            f"{start.date() if start is not None else '-inf'} to "
            f"{end.date() if end is not None else 'inf'}"
        )
    return filtered


def _export_station_csv(series: pd.Series, path: Path) -> None:
    """Write a single-station time-series to CSV."""
    frame = pd.DataFrame({"date": series.index.strftime("%Y-%m-%d"), "displacement_m": series.values})
    frame.to_csv(path, index=False, float_format="%.6f")


def _plot_station(
    station: Station,
    series_abs: pd.Series,
    series_rel: pd.Series,
    orbit_label: str,
    output_path: Path,
) -> None:
    """Render absolute/relative displacement plot for a station."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].plot(series_abs.index, series_abs.values, color="tab:blue")
    axes[0].set_title(f"{orbit_label} | {station.name} absolute LOS displacement")
    axes[0].set_ylabel("Displacement (m)")
    axes[0].grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    axes[1].plot(series_rel.index, series_rel.values, color="tab:orange")
    axes[1].set_title(f"{orbit_label} | {station.name} relative to ST01")
    axes[1].set_ylabel("Displacement (m)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def process_orbit(
    label: str,
    h5_path: Path,
    output_dir: Path,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single orbit direction and export CSV/plot artefacts."""
    frame_abs = _read_timeseries(h5_path, STATIONS)
    frame_abs = _filter_time_range(frame_abs, start, end)
    frame_rel = _relative_to_reference(frame_abs, "ST01")

    orbit_tag = label.lower()

    for station in STATIONS:
        abs_csv_path = output_dir / f"{station.name}_{orbit_tag}_absolute.csv"
        rel_csv_path = output_dir / f"{station.name}_{orbit_tag}_relative_to_ST01.csv"
        plot_path = output_dir / f"{station.name}_{orbit_tag}_timeseries.png"

        series_abs = frame_abs[station.name]
        series_rel = frame_rel[station.name]

        _export_station_csv(series_abs, abs_csv_path)
        _export_station_csv(series_rel, rel_csv_path)
        _plot_station(station, series_abs, series_rel, label.capitalize(), plot_path)

    return frame_abs, frame_rel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ANSI station LOS time-series from MintPy outputs."
    )
    parser.add_argument(
        "--ascending",
        type=Path,
        required=True,
        help="Path to ascending geo_timeseries_*.h5 file",
    )
    parser.add_argument(
        "--descending",
        type=Path,
        required=True,
        help="Path to descending geo_timeseries_*.h5 file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory for outputs; used only if orbit specific output "
            "paths are not provided (default: current directory)"
        ),
    )
    parser.add_argument(
        "--ascending-output-dir",
        type=Path,
        default=None,
        help="Directory to store ascending orbit outputs (created if missing)",
    )
    parser.add_argument(
        "--descending-output-dir",
        type=Path,
        default=None,
        help="Directory to store descending orbit outputs (created if missing)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Inclusive start date (YYYY-MM-DD or YYYYMMDD) for extraction",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Inclusive end date (YYYY-MM-DD or YYYYMMDD) for extraction",
    )
    args = parser.parse_args()

    def parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        try:
            return pd.to_datetime(value, format="%Y-%m-%d", utc=False)
        except ValueError:
            try:
                return pd.to_datetime(value, format="%Y%m%d", utc=False)
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse date '{value}'. Use YYYY-MM-DD or YYYYMMDD."
                ) from exc

    start_dt = parse_date(args.start_date)
    end_dt = parse_date(args.end_date)
    if start_dt is not None:
        start_dt = start_dt.normalize()
    if end_dt is not None:
        end_dt = end_dt.normalize()
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        raise ValueError("Start date must be on or before end date.")

    if args.output_dir is None and args.ascending_output_dir is None and args.descending_output_dir is None:
        # Default to current directory when nothing else is provided.
        base_output = Path(".")
    else:
        base_output = args.output_dir

    if args.ascending_output_dir is not None:
        asc_dir = args.ascending_output_dir
    elif base_output is not None:
        asc_dir = base_output / "ascending"
    else:
        asc_dir = Path(".") / "ascending"

    if args.descending_output_dir is not None:
        dsc_dir = args.descending_output_dir
    elif base_output is not None:
        dsc_dir = base_output / "descending"
    else:
        dsc_dir = Path(".") / "descending"

    asc_dir.mkdir(parents=True, exist_ok=True)
    dsc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing ascending stack: {args.ascending}")
    asc_abs, asc_rel = process_orbit(
        "ascending", args.ascending, asc_dir, start_dt, end_dt
    )
    print(f"Processing descending stack: {args.descending}")
    desc_abs, desc_rel = process_orbit(
        "descending", args.descending, dsc_dir, start_dt, end_dt
    )

    summary = {
        "ascending": {
            "dates": (asc_abs.index.min(), asc_abs.index.max(), asc_abs.shape[0]),
            "stations": asc_abs.columns.tolist(),
        },
        "descending": {
            "dates": (desc_abs.index.min(), desc_abs.index.max(), desc_abs.shape[0]),
            "stations": desc_abs.columns.tolist(),
        },
    }

    print("\nSummary")
    for orbit, info in summary.items():
        start, end, count = info["dates"]
        print(
            f"  {orbit.capitalize():<10} {count:>4} epochs between "
            f"{start.date()} and {end.date()} for {len(info['stations'])} stations."
        )


if __name__ == "__main__":
    main()
