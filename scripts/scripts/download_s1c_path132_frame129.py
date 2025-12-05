#!/usr/bin/env python3
"""
Download Sentinel-1C IW SLC scenes for relative orbit 132, frame 129 (ascending)
into a target directory.

Defaults:
  - platform: Sentinel-1C
  - beamMode: IW
  - processingLevel: SLC
  - relativeOrbit: 132
  - frame: 129
  - date range: 2024-01-01 to 2025-06-30
  - target directory: /mnt/beegfs/procesados/sendu/garrox/ASC/ASCENDING_132

Credentials are read from ~/.netrc (machine urs.earthdata.nasa.gov).
Requires the asf_search package: `pip install asf_search`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple
from netrc import netrc

import asf_search as asf


def get_credentials() -> Tuple[str, str]:
    auth = netrc().authenticators("urs.earthdata.nasa.gov")
    if not auth:
        raise RuntimeError("No Earthdata credentials in ~/.netrc for urs.earthdata.nasa.gov")
    username, _, password = auth
    return username, password


def make_session() -> asf.ASFSession:
    user, pwd = get_credentials()
    session = asf.ASFSession()
    session.auth_with_creds(user, pwd)
    return session


def search_products(
    session: asf.ASFSession,
    start: str,
    end: str,
    orbit: int,
    frame: int,
) -> List[asf.ASFProduct]:
    opts = asf.ASFSearchOptions(
        platform=["Sentinel-1C"],
        beamMode=["IW"],
        processingLevel=["SLC"],
        flightDirection="ASCENDING",
        relativeOrbit=[orbit],
        frame=[frame],
        start=start,
        end=end,
    )
    # asf.search does not take session kwarg in this installed version; rely on ~/.netrc
    return list(asf.search(opts=opts))


def download(products: Iterable[asf.ASFProduct], target_dir: Path, session: asf.ASFSession, overwrite: bool) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for prod in products:
        dest = target_dir / prod.properties["fileID"]
        if dest.exists() and not overwrite:
            print(f"[skip] {dest.name} exists")
            continue
        print(f"[down] {dest.name}")
        prod.download(path=str(target_dir), session=session)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Sentinel-1C IW SLC for orbit 132 frame 129.")
    p.add_argument("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD).")
    p.add_argument("--end-date", default="2025-06-30", help="End date (YYYY-MM-DD).")
    p.add_argument(
        "--target",
        default="/mnt/beegfs/procesados/sendu/garrox/ASC/ASCENDING_132",
        help="Destination directory for ZIP files.",
    )
    p.add_argument("--overwrite", action="store_true", help="Re-download even if file exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session = make_session()
    products = search_products(session, args.start_date, args.end_date, orbit=132, frame=129)
    print(f"Found {len(products)} products for S1C orbit 132 frame 129 in [{args.start_date}, {args.end_date}]")
    download(products, Path(args.target), session, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
