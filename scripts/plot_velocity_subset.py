#!/usr/bin/env python3
"""
Plot MintPy average deformation rate within a polygon defined in a KML file.
"""

from __future__ import annotations

import argparse
import pathlib
from xml.etree import ElementTree as ET

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path


KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MintPy velocity within a KML polygon and plot it."
    )
    parser.add_argument(
        "--velocity",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/descending/"
            "mintpy20240723/geo/geo_velocity.h5"
        ),
        help="Path to geo_velocity.h5 (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        default="velocity",
        help="Dataset name inside the HDF5 file (default: %(default)s)",
    )
    parser.add_argument(
        "--kml",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/ascending/"
            "mintpy20240723/geo/dam_con_dam.kml"
        ),
        help="KML file containing the polygon of interest (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="dam_con_dam_velocity_des.png",
        help="Output PNG file (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Figure resolution in DPI (default: %(default)s)"
    )
    return parser.parse_args()


def read_polygons(kml_path: pathlib.Path) -> list[np.ndarray]:
    tree = ET.parse(kml_path)
    polygons: list[np.ndarray] = []
    for coord_node in tree.findall(".//kml:Polygon//kml:coordinates", namespaces=KML_NS):
        raw_text = (coord_node.text or "").strip()
        if not raw_text:
            continue
        coords = []
        for chunk in raw_text.split():
            parts = chunk.split(",")
            if len(parts) < 2:
                continue
            lon, lat = map(float, parts[:2])
            coords.append((lon, lat))
        if len(coords) >= 3:
            polygons.append(np.asarray(coords))
    if not polygons:
        raise ValueError(f"No polygon coordinates found in {kml_path}")
    return polygons


def load_velocity(
    velocity_path: pathlib.Path, dataset: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with h5py.File(velocity_path, "r") as f:
        vel = f[dataset][:]
        x_first = float(f.attrs["X_FIRST"])
        x_step = float(f.attrs["X_STEP"])
        width = int(f.attrs["WIDTH"])
        y_first = float(f.attrs["Y_FIRST"])
        y_step = float(f.attrs["Y_STEP"])
        length = int(f.attrs["LENGTH"])
        unit = f.attrs.get("UNIT", "m/year")

    lon = x_first + np.arange(width) * x_step
    lat = y_first + np.arange(length) * y_step
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return vel, lon_grid, lat_grid, unit


def compute_mask(polygons: list[np.ndarray], lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    mask = np.zeros(lon_grid.shape, dtype=bool)
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    for poly in polygons:
        path = Path(poly[:, :2])
        contains = path.contains_points(points)
        mask |= contains.reshape(lon_grid.shape)
    return mask


def main() -> None:
    args = parse_args()

    velocity_path = pathlib.Path(args.velocity).expanduser()
    kml_path = pathlib.Path(args.kml).expanduser()
    output_path = pathlib.Path(args.output).expanduser()

    polygons = read_polygons(kml_path)
    vel, lon_grid, lat_grid, unit = load_velocity(velocity_path, args.dataset)
    mask = compute_mask(polygons, lon_grid, lat_grid)

    if not np.any(mask):
        raise RuntimeError("Polygon does not overlap the velocity grid.")

    data_in_poly = vel[mask]
    valid = data_in_poly[np.isfinite(data_in_poly)]
    if valid.size == 0:
        raise RuntimeError("No finite velocity values inside the polygon.")

    vmin, vmax = np.nanpercentile(valid, (2, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(valid)), float(np.nanmax(valid))

    masked_vel = np.ma.masked_where(~mask, vel)

    all_coords = np.vstack(polygons)
    lon_min, lon_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    lat_min, lat_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    pad_lon = max((lon_max - lon_min) * 0.05, 0.0005)
    pad_lat = max((lat_max - lat_min) * 0.05, 0.0005)

    fig, ax = plt.subplots(figsize=(6, 6))
    pcm = ax.pcolormesh(
        lon_grid,
        lat_grid,
        masked_vel,
        cmap="RdYlBu_r",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    for poly in polygons:
        ax.plot(poly[:, 0], poly[:, 1], color="k", linewidth=0.8)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.06, shrink=1.0)
    cbar.set_label(f"Velocity ({unit})")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Average deformation rate")
    ax.set_aspect("equal")
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
