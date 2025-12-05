#!/usr/bin/env python3
"""
Decompose ascending/descending MintPy velocities into vertical and east components inside a polygon.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, Tuple
from xml.etree import ElementTree as ET

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import griddata

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose ascending & descending MintPy velocities into vertical and east components."
    )
    parser.add_argument(
        "--asc-velocity",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/ascending/"
            "mintpy20240723/geo/geo_velocity.h5"
        ),
        help="Ascending geo_velocity.h5 path (default: %(default)s)",
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
        "--desc-velocity",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/descending/"
            "mintpy20240723/geo/geo_velocity.h5"
        ),
        help="Descending geo_velocity.h5 path (default: %(default)s)",
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
        "--dataset",
        default="velocity",
        help="Velocity dataset name inside HDF5 (default: %(default)s)",
    )
    parser.add_argument(
        "--kml",
        default=(
            "/mnt/beegfs/procesados/sendu/riotinto/ascending/"
            "mintpy20240723/geo/dam_con_dam.kml"
        ),
        help="KML file containing polygon of interest (default: %(default)s)",
    )
    parser.add_argument(
        "--output-prefix",
        default="dam_con_dam",
        help="Prefix for output figures (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Figure resolution in DPI (default: %(default)s)"
    )
    return parser.parse_args()


def read_polygons(kml_path: pathlib.Path) -> list[np.ndarray]:
    tree = ET.parse(kml_path)
    polygons: list[np.ndarray] = []
    for node in tree.findall(".//kml:Polygon//kml:coordinates", namespaces=KML_NS):
        raw = (node.text or "").strip()
        if not raw:
            continue
        coords: list[Tuple[float, float]] = []
        for chunk in raw.split():
            parts = chunk.split(",")
            if len(parts) < 2:
                continue
            lon, lat = map(float, parts[:2])
            coords.append((lon, lat))
        if len(coords) >= 3:
            polygons.append(np.asarray(coords))
    if not polygons:
        raise ValueError(f"No polygon found in {kml_path}")
    return polygons


def load_track_data(
    velocity_path: pathlib.Path,
    geometry_path: pathlib.Path,
    dataset: str,
) -> dict[str, np.ndarray | str]:
    with h5py.File(velocity_path, "r") as vf:
        vel = vf[dataset][:].astype(np.float32)
        unit = vf.attrs.get("UNIT", "m/year")
    with h5py.File(geometry_path, "r") as gf:
        inc = gf["incidenceAngle"][:].astype(np.float32)
        az = gf["azimuthAngle"][:].astype(np.float32)
        lon = gf["longitude"][:].astype(np.float64)
        lat = gf["latitude"][:].astype(np.float64)
    return {"velocity": vel, "inc": inc, "az": az, "lon": lon, "lat": lat, "unit": unit}


def polygon_mask(lon: np.ndarray, lat: np.ndarray, polygons: Iterable[np.ndarray]) -> np.ndarray:
    points = np.column_stack((lon.ravel(), lat.ravel()))
    mask = np.zeros(lon.size, dtype=bool)
    for poly in polygons:
        path = Path(poly[:, :2])
        mask |= path.contains_points(points)
    return mask.reshape(lon.shape)


def crop_to_mask(arrays: dict[str, np.ndarray], mask: np.ndarray, pad: int = 5) -> dict[str, np.ndarray]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Polygon does not overlap the provided grid.")
    r0 = max(rows[0] - pad, 0)
    r1 = min(rows[-1] + pad + 1, mask.shape[0])
    c0 = max(cols[0] - pad, 0)
    c1 = min(cols[-1] + pad + 1, mask.shape[1])
    sliced = {k: v[r0:r1, c0:c1] for k, v in arrays.items()}
    sliced["mask"] = mask[r0:r1, c0:c1]
    return sliced


def interpolate_to_grid(
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    src_val: np.ndarray,
    dst_lon: np.ndarray,
    dst_lat: np.ndarray,
) -> np.ndarray:
    points = np.column_stack((src_lon.ravel(), src_lat.ravel()))
    values = src_val.ravel()
    valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
    if not np.any(valid):
        raise ValueError("No valid samples available for interpolation.")
    points = points[valid]
    values = values[valid]
    targets = np.column_stack((dst_lon.ravel(), dst_lat.ravel()))
    interp = griddata(points, values, targets, method="linear")
    if np.any(np.isnan(interp)):
        nearest = griddata(points, values, targets[np.isnan(interp)], method="nearest")
        interp[np.isnan(interp)] = nearest
    return interp.reshape(dst_lon.shape)


def look_components(inc_deg: np.ndarray, az_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inc = np.deg2rad(inc_deg)
    az = np.deg2rad(az_deg)
    sin_inc = np.sin(inc)
    u_e = -sin_inc * np.sin(az)
    u_u = np.cos(inc)
    return u_e, u_u


def solve_east_up(
    v_asc: np.ndarray,
    v_desc: np.ndarray,
    u_e_asc: np.ndarray,
    u_u_asc: np.ndarray,
    u_e_desc: np.ndarray,
    u_u_desc: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    det = u_e_asc * u_u_desc - u_e_desc * u_u_asc
    valid = (
        mask
        & np.isfinite(v_asc)
        & np.isfinite(v_desc)
        & np.isfinite(u_e_asc)
        & np.isfinite(u_u_asc)
        & np.isfinite(u_e_desc)
        & np.isfinite(u_u_desc)
        & (np.abs(det) > 1e-8)
    )
    east = np.full_like(v_asc, np.nan, dtype=np.float32)
    up = np.full_like(v_asc, np.nan, dtype=np.float32)
    east[valid] = (v_asc[valid] * u_u_desc[valid] - v_desc[valid] * u_u_asc[valid]) / det[valid]
    up[valid] = (u_e_asc[valid] * v_desc[valid] - u_e_desc[valid] * v_asc[valid]) / det[valid]
    return east, up


def component_limits(data: np.ndarray) -> tuple[float, float]:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise ValueError("No valid pixels inside the polygon.")
    vmin, vmax = np.nanpercentile(valid, (2, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(valid))
        vmax = float(np.nanmax(valid))
    if vmin == vmax:
        delta = abs(vmin) if vmin != 0 else 1.0
        vmin -= delta
        vmax += delta
    return vmin, vmax


def plot_component(
    data: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    polygons: Iterable[np.ndarray],
    title: str,
    unit: str,
    output: pathlib.Path,
    dpi: int,
):
    masked = np.ma.masked_invalid(data)
    vmin, vmax = component_limits(masked.filled(np.nan))
    lon_all = np.hstack([poly[:, 0] for poly in polygons])
    lat_all = np.hstack([poly[:, 1] for poly in polygons])
    pad_lon = max((lon_all.max() - lon_all.min()) * 0.05, 0.0005)
    pad_lat = max((lat_all.max() - lat_all.min()) * 0.05, 0.0005)

    fig, ax = plt.subplots(figsize=(6, 6))
    triang = mtri.Triangulation(lon.ravel(), lat.ravel())
    pcm = ax.tripcolor(
        triang,
        masked.ravel(),
        cmap="RdBu_r",
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )
    for poly in polygons:
        ax.plot(poly[:, 0], poly[:, 1], color="k", linewidth=0.8)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.06, shrink=1.0)
    cbar.set_label(f"{title} ({unit})")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(lon_all.min() - pad_lon, lon_all.max() + pad_lon)
    ax.set_ylim(lat_all.min() - pad_lat, lat_all.max() + pad_lat)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    args = parse_args()

    asc_vel_path = pathlib.Path(args.asc_velocity).expanduser()
    asc_geom_path = pathlib.Path(args.asc_geometry).expanduser()
    desc_vel_path = pathlib.Path(args.desc_velocity).expanduser()
    desc_geom_path = pathlib.Path(args.desc_geometry).expanduser()
    kml_path = pathlib.Path(args.kml).expanduser()
    output_prefix = pathlib.Path(args.output_prefix).expanduser()

    polygons = read_polygons(kml_path)
    asc = load_track_data(asc_vel_path, asc_geom_path, args.dataset)
    mask = polygon_mask(asc["lon"], asc["lat"], polygons)
    asc_sub = crop_to_mask(
        {"velocity": asc["velocity"], "inc": asc["inc"], "az": asc["az"], "lon": asc["lon"], "lat": asc["lat"]},
        mask,
    )

    desc = load_track_data(desc_vel_path, desc_geom_path, args.dataset)
    desc_vel_on_asc = interpolate_to_grid(desc["lon"], desc["lat"], desc["velocity"], asc_sub["lon"], asc_sub["lat"])
    desc_inc_on_asc = interpolate_to_grid(desc["lon"], desc["lat"], desc["inc"], asc_sub["lon"], asc_sub["lat"])
    desc_az_on_asc = interpolate_to_grid(desc["lon"], desc["lat"], desc["az"], asc_sub["lon"], asc_sub["lat"])

    u_e_asc, u_u_asc = look_components(asc_sub["inc"], asc_sub["az"])
    u_e_desc, u_u_desc = look_components(desc_inc_on_asc, desc_az_on_asc)

    east, up = solve_east_up(
        asc_sub["velocity"],
        desc_vel_on_asc,
        u_e_asc,
        u_u_asc,
        u_e_desc,
        u_u_desc,
        asc_sub["mask"],
    )

    east = np.where(asc_sub["mask"], east, np.nan)
    up = np.where(asc_sub["mask"], up, np.nan)
    unit = asc["unit"]
    prefix_str = str(output_prefix)
    vertical_path = pathlib.Path(f"{prefix_str}_vertical.png")
    east_path = pathlib.Path(f"{prefix_str}_east.png")

    plot_component(
        up,
        asc_sub["lon"],
        asc_sub["lat"],
        polygons,
        "Vertical deformation rate",
        unit,
        pathlib.Path(vertical_path),
        args.dpi,
    )
    plot_component(
        east,
        asc_sub["lon"],
        asc_sub["lat"],
        polygons,
        "East deformation rate",
        unit,
        pathlib.Path(east_path),
        args.dpi,
    )


if __name__ == "__main__":
    main()
