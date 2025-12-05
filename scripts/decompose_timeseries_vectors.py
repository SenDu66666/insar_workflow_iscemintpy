#!/usr/bin/env python3
"""
Decompose MintPy ascending/descending point time series into vertical and east components.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Iterable, Tuple
from xml.etree import ElementTree as ET

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose MintPy point time series into vertical and east components."
    )
    parser.add_argument(
        "--asc-ts-dir",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/ascending/"
            "mintpy20240723/geo/ANSI32TS_ASC"
        ),
        help="Directory with ascending CSV time series (default: %(default)s)",
    )
    parser.add_argument(
        "--desc-ts-dir",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/descending/"
            "mintpy20240723/geo/ANSI32TS_DSC"
        ),
        help="Directory with descending CSV time series (default: %(default)s)",
    )
    parser.add_argument(
        "--asc-geometry",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/ascending/"
            "mintpy20240723/geo/geo_geometryRadar.h5"
        ),
        help="Ascending geo_geometryRadar.h5 path (default: %(default)s)",
    )
    parser.add_argument(
        "--desc-geometry",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/descending/"
            "mintpy20240723/geo/geo_geometryRadar.h5"
        ),
        help="Descending geo_geometryRadar.h5 path (default: %(default)s)",
    )
    parser.add_argument(
        "--kml",
        default=str(pathlib.Path(__file__).with_name("ansi_stations.kml")),
        help="KML with station coordinates (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/ascending/"
            "mintpy20240723/geo/ANSI32TS_VEC"
        ),
        help="Directory to store decomposed time series (default: %(default)s)",
    )
    parser.add_argument(
        "--reference",
        default="ST01",
        help="Station to use as relative reference (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Figure resolution (default: %(default)s)"
    )
    return parser.parse_args()


def read_station_coords(kml_path: pathlib.Path) -> Dict[str, Tuple[float, float]]:
    tree = ET.parse(kml_path)
    coords: Dict[str, Tuple[float, float]] = {}
    for placemark in tree.findall(".//kml:Placemark", namespaces=KML_NS):
        name_node = placemark.find("kml:name", namespaces=KML_NS)
        coord_node = placemark.find(".//kml:coordinates", namespaces=KML_NS)
        if name_node is None or coord_node is None:
            continue
        name = name_node.text.strip()
        lon_lat_alt = coord_node.text.strip().split(",")
        if len(lon_lat_alt) < 2:
            continue
        lon, lat = map(float, lon_lat_alt[:2])
        coords[name] = (lon, lat)
    if not coords:
        raise ValueError(f"No station coordinates found in {kml_path}")
    return coords


def load_geometry(path: pathlib.Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        geom = {
            "lon": f["longitude"][:],
            "lat": f["latitude"][:],
            "inc": f["incidenceAngle"][:],
            "az": f["azimuthAngle"][:],
        }
    return geom


def sample_geometry(geom: Dict[str, np.ndarray], lon_pt: float, lat_pt: float) -> Tuple[float, float]:
    dist = (geom["lon"] - lon_pt) ** 2 + (geom["lat"] - lat_pt) ** 2
    if np.all(~np.isfinite(dist)):
        raise ValueError("Geometry grid does not contain finite coordinates.")
    idx = np.nanargmin(dist)
    row, col = np.unravel_index(idx, geom["lon"].shape)
    inc = float(geom["inc"][row, col])
    az = float(geom["az"][row, col])
    if not np.isfinite(inc) or not np.isfinite(az):
        raise ValueError("Sampled incidence or azimuth is NaN.")
    return inc, az


def load_timeseries(csv_path: pathlib.Path) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date")
    return pd.DatetimeIndex(df["date"].values), df["displacement_m"].to_numpy(dtype=np.float64)


def interpolate_series(
    src_dates: pd.DatetimeIndex, src_values: np.ndarray, target_dates: pd.DatetimeIndex
) -> np.ndarray:
    if len(src_dates) == 0:
        return np.full(len(target_dates), np.nan)
    src_ord = src_dates.view("int64") / 1e9  # seconds
    tgt_ord = target_dates.view("int64") / 1e9
    order = np.argsort(src_ord)
    src_ord = src_ord[order]
    src_vals = src_values[order]
    interp = np.interp(tgt_ord, src_ord, src_vals, left=np.nan, right=np.nan)
    outside = (tgt_ord < src_ord[0]) | (tgt_ord > src_ord[-1])
    interp[outside] = np.nan
    return interp


def look_components(inc_deg: float, az_deg: float) -> Tuple[float, float]:
    inc = np.deg2rad(inc_deg)
    az = np.deg2rad(az_deg)
    sin_inc = np.sin(inc)
    u_e = -sin_inc * np.sin(az)
    u_u = np.cos(inc)
    return float(u_e), float(u_u)


def solve_east_up(
    v_asc: np.ndarray,
    v_desc: np.ndarray,
    u_e_asc: float,
    u_u_asc: float,
    u_e_desc: float,
    u_u_desc: float,
) -> Tuple[np.ndarray, np.ndarray]:
    det = u_e_asc * u_u_desc - u_e_desc * u_u_asc
    east = np.full_like(v_desc, np.nan, dtype=np.float64)
    up = np.full_like(v_desc, np.nan, dtype=np.float64)
    if abs(det) < 1e-8:
        return east, up
    mask = np.isfinite(v_asc) & np.isfinite(v_desc)
    east[mask] = (v_asc[mask] * u_u_desc - v_desc[mask] * u_u_asc) / det
    up[mask] = (u_e_asc * v_desc[mask] - u_e_desc * v_asc[mask]) / det
    return east, up


def ensure_output_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_stations(asc_dir: pathlib.Path, desc_dir: pathlib.Path) -> list[str]:
    asc = {p.name.split("_")[0] for p in asc_dir.glob("ST*_ascending_absolute.csv")}
    desc = {p.name.split("_")[0] for p in desc_dir.glob("ST*_descending_absolute.csv")}
    stations = sorted(asc & desc)
    if not stations:
        raise ValueError("No stations found in the provided directories.")
    return stations


def write_csv(dates: pd.DatetimeIndex, values: np.ndarray, path: pathlib.Path) -> None:
    df = pd.DataFrame({"date": dates, "deformation_m": values})
    df.to_csv(path, index=False)


def plot_timeseries(
    dates: pd.DatetimeIndex,
    absolute: np.ndarray,
    relative: np.ndarray,
    title: str,
    ylabel: str,
    output: pathlib.Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, absolute, label="Absolute", color="tab:blue")
    ax.plot(dates, relative, label="Relative to ST01", color="tab:orange", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    asc_ts_dir = pathlib.Path(args.asc_ts_dir)
    desc_ts_dir = pathlib.Path(args.desc_ts_dir)
    asc_geom_path = pathlib.Path(args.asc_geometry)
    desc_geom_path = pathlib.Path(args.desc_geometry)
    kml_path = pathlib.Path(args.kml)
    output_dir = pathlib.Path(args.output_dir)
    reference = args.reference.upper()

    ensure_output_dir(output_dir)

    station_coords = read_station_coords(kml_path)
    stations = discover_stations(asc_ts_dir, desc_ts_dir)
    if reference not in stations:
        raise ValueError(f"Reference station {reference} not found among stations {stations}")

    asc_geom = load_geometry(asc_geom_path)
    desc_geom = load_geometry(desc_geom_path)

    master_dates: pd.DatetimeIndex | None = None
    results: Dict[str, Dict[str, np.ndarray]] = {}

    for station in stations:
        desc_abs_path = desc_ts_dir / f"{station}_descending_absolute.csv"
        asc_abs_path = asc_ts_dir / f"{station}_ascending_absolute.csv"

        desc_dates, desc_abs = load_timeseries(desc_abs_path)
        if master_dates is None:
            master_dates = desc_dates
        else:
            if len(master_dates) != len(desc_dates) or not np.array_equal(master_dates.values, desc_dates.values):
                raise ValueError("Descending timelines differ between stations.")

        asc_dates, asc_abs = load_timeseries(asc_abs_path)
        asc_interp = interpolate_series(asc_dates, asc_abs, master_dates)

        if station not in station_coords:
            raise ValueError(f"Station {station} missing coordinates in {kml_path}")
        lon_pt, lat_pt = station_coords[station]
        inc_asc, az_asc = sample_geometry(asc_geom, lon_pt, lat_pt)
        inc_desc, az_desc = sample_geometry(desc_geom, lon_pt, lat_pt)
        u_e_asc, u_u_asc = look_components(inc_asc, az_asc)
        u_e_desc, u_u_desc = look_components(inc_desc, az_desc)

        east_abs, up_abs = solve_east_up(
            asc_interp,
            desc_abs,
            u_e_asc,
            u_u_asc,
            u_e_desc,
            u_u_desc,
        )

        results[station] = {
            "vertical_abs": up_abs,
            "east_abs": east_abs,
        }

    assert master_dates is not None
    ref_vertical = results[reference]["vertical_abs"]
    ref_east = results[reference]["east_abs"]

    for station, data in results.items():
        data["vertical_rel"] = data["vertical_abs"] - ref_vertical
        data["east_rel"] = data["east_abs"] - ref_east

        vert_abs_path = output_dir / f"{station}_vertical_absolute.csv"
        vert_rel_path = output_dir / f"{station}_vertical_relative_to_{reference}.csv"
        east_abs_path = output_dir / f"{station}_east_absolute.csv"
        east_rel_path = output_dir / f"{station}_east_relative_to_{reference}.csv"
        vert_png = output_dir / f"{station}_vertical_timeseries.png"
        east_png = output_dir / f"{station}_east_timeseries.png"

        write_csv(master_dates, data["vertical_abs"], vert_abs_path)
        write_csv(master_dates, data["vertical_rel"], vert_rel_path)
        write_csv(master_dates, data["east_abs"], east_abs_path)
        write_csv(master_dates, data["east_rel"], east_rel_path)

        plot_timeseries(
            master_dates,
            data["vertical_abs"],
            data["vertical_rel"],
            f"{station} vertical deformation",
            "Deformation (m)",
            vert_png,
            args.dpi,
        )
        plot_timeseries(
            master_dates,
            data["east_abs"],
            data["east_rel"],
            f"{station} east deformation",
            "Deformation (m)",
            east_png,
            args.dpi,
        )

    print(f"Saved decomposed time series for {len(stations)} stations to {output_dir}")


if __name__ == "__main__":
    main()
