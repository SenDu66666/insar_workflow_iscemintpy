#!/usr/bin/env python3
"""
Utility to keep Sentinel-1 SLC archives in sync for specified ascending relative orbits.

Workflow
--------
1. Query ASF Search for Sentinel-1 IW SLC products intersecting the provided AOI/WKT.
2. Compare the remote list with two local roots:
     - A read-only archive (e.g. /mnt/SLC_synology/canarias.s1/ASCENDING_<orbit>)
     - The writable target root (e.g. /mnt/beegfs/procesados/sendu/ASCENDING_<orbit>)
3. Download any missing ZIP files with multiple concurrent workers.

Features
--------
* Uses Earthdata credentials stored in ~/.netrc (machine urs.earthdata.nasa.gov).
* Generates orbit-specific destination folders automatically (ASCENDING_<orbit>).
* Retries downloads with exponential back-off and resumes partial files.
* Maintains per-download cookie jars to avoid race conditions when running in parallel.
* Supports dry-run mode to quickly list which scenes would be fetched.

Example
-------
    python download_missing_s1.py --orbits 60 89 162 --threads 3 --dry-run
    nohup python download_missing_s1.py --orbits 60 89 162 --threads 3 > mintpy_s1.log 2>&1 &
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import time
import socket
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from netrc import netrc

try:
    import asf_search as asf
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'asf_search'. Activate the mintpy2025 environment "
        "or install asf-search before running this script."
    ) from exc


DEFAULT_WKT = (
    "POLYGON(("
    "-19.5 26.5, "
    "-19.5 29.5, "
    "-15.0 29.5, "
    "-15.0 26.5, "
    "-19.5 26.5"
    "))"
)


@dataclass
class DownloadJob:
    orbit: int
    scene: str
    url: str
    destination: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download missing Sentinel-1 IW SLC scenes for specified ascending orbits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--orbits",
        nargs="+",
        type=int,
        required=True,
        help="Sentinel-1 relative orbit (path) numbers to process, e.g. 60 89 162.",
    )
    parser.add_argument(
        "--start-date",
        default="2018-01-01",
        help="ISO date (YYYY-MM-DD) for start of search window (UTC).",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="ISO date (YYYY-MM-DD) for end of search window (UTC).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Maximum number of concurrent downloads.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of attempts per scene before giving up.",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=int,
        default=60,
        help="Initial back-off between retries; doubles after each failure.",
    )
    parser.add_argument(
        "--direction",
        choices=["ASCENDING", "DESCENDING"],
        default="ASCENDING",
        help="Orbit direction to process. Run the script twice for both directions.",
    )
    parser.add_argument(
        "--wkt",
        default=DEFAULT_WKT,
        help=(
            "WKT polygon/geometry used for ASF spatial filtering. "
            "Defaults to a bounding box covering the Canary Islands."
        ),
    )
    parser.add_argument(
        "--existing-root",
        default="/mnt/SLC_synology/canarias.s1",
        help="Read-only archive root with existing ASCENDING_<orbit> folders.",
    )
    parser.add_argument(
        "--target-root",
        default="/mnt/beegfs/procesados/sendu",
        help="Writable root where new ASCENDING_<orbit> folders will be created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report missing scenes; do not download.",
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=300,
        help="HTTP timeout passed to wget in seconds.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable TLS certificate verification for wget (--no-check-certificate).",
    )
    return parser.parse_args()


def get_earthdata_credentials() -> Tuple[str, str]:
    auth = netrc().authenticators("urs.earthdata.nasa.gov")
    if not auth:
        raise RuntimeError("No Earthdata credentials found in ~/.netrc for urs.earthdata.nasa.gov")
    username, _, password = auth
    return username, password


def build_asf_session(username: str, password: str) -> asf.ASFSession:
    session = asf.ASFSession()
    session.auth_with_creds(username, password)
    return session


def patch_dns() -> None:
    """Work around local DNS resolution issues for specific NASA hosts."""
    overrides = {
        "cmr.earthdata.nasa.gov": "13.224.83.47",
    }

    original_getaddrinfo = socket.getaddrinfo

    def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        mapped_host = overrides.get(host, host)
        return original_getaddrinfo(mapped_host, port, family, type, proto, flags)

    socket.getaddrinfo = custom_getaddrinfo


def collect_remote_scenes(
    session: asf.ASFSession,
    orbits: Iterable[int],
    wkt: str,
    start_date: str,
    end_date: str,
    direction: str,
) -> Dict[int, Dict[str, asf.ASFProduct]]:
    orbit_set = set(orbits)
    mapping: Dict[int, Dict[str, asf.ASFProduct]] = {orbit: {} for orbit in orbit_set}

    opts = asf.ASFSearchOptions(
        platform=["Sentinel-1A", "Sentinel-1B"],
        beamMode=["IW"],
        flightDirection=direction,
        processingLevel=["SLC"],
        intersectsWith=wkt,
        start=f"{start_date}T00:00:00Z",
        end=f"{end_date}T23:59:59Z",
        maxResults=2000,
        session=session,
    )
    results = asf.search(opts=opts)

    for product in results:
        scene = product.properties["sceneName"]
        orbit = product.properties.get("pathNumber")
        if orbit in mapping:
            mapping[orbit][scene] = product

    return mapping


def collect_local_scenes(root: Path, orbit: int, direction: str) -> Set[str]:
    orbit_dir = root / f"{direction}_{orbit}"
    if not orbit_dir.exists():
        return set()
    return {
        path.stem
        for path in orbit_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".zip"
    }


def prepare_download_jobs(
    orbit: int,
    scenes: Iterable[str],
    direction: str,
    target_root: Path,
) -> List[DownloadJob]:
    jobs: List[DownloadJob] = []
    dest_dir = target_root / f"{direction}_{orbit}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for scene in sorted(scenes):
        target_file = dest_dir / f"{scene}.zip"
        if target_file.exists():
            continue
        mission = scene[:3]
        subdir = "SA" if mission == "S1A" else "SB"
        url = f"https://datapool.asf.alaska.edu/SLC/{subdir}/{scene}.zip"
        jobs.append(
            DownloadJob(
                orbit=orbit,
                scene=scene,
                url=url,
                destination=target_file,
            )
        )
    return jobs


def run_wget(
    job: DownloadJob,
    username: str,
    password: str,
    timeout: int,
    disable_tls_verify: bool,
    attempt: int,
) -> subprocess.CompletedProcess:
    cookies_fd, cookies_path = tempfile.mkstemp(prefix=f"{job.scene}_", suffix=".cookies")
    os.close(cookies_fd)

    cmd = [
        "wget",
        "--auth-no-challenge",
        "--keep-session-cookies",
        f"--load-cookies={cookies_path}",
        f"--save-cookies={cookies_path}",
        "--continue",
        "--no-verbose",
        "--show-progress",
        "--progress=dot:giga",
        f"--timeout={timeout}",
        f"--read-timeout={timeout}",
        "--waitretry=30",
        "--tries=10",
        "--retry-connrefused",
        "--user",
        username,
        "--password",
        password,
        "-O",
        str(job.destination),
        job.url,
    ]

    if disable_tls_verify:
        cmd.insert(1, "--no-check-certificate")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        # We intentionally retain cookies on success for incremental resumes.
        if result := locals().get("result"):
            if result.returncode == 0:
                shutil.move(cookies_path, f"{cookies_path}.last")
            else:
                os.unlink(cookies_path)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def download_job(
    job: DownloadJob,
    username: str,
    password: str,
    max_retries: int,
    timeout: int,
    disable_tls_verify: bool,
    initial_backoff: float,
) -> Tuple[str, bool, Optional[str]]:
    if job.destination.exists():
        return (job.scene, True, "already-present")

    backoff = float(initial_backoff)
    if backoff <= 0:
        backoff = float(timeout)

    attempt = 0

    while attempt <= max_retries:
        attempt += 1
        try:
            run_wget(
                job=job,
                username=username,
                password=password,
                timeout=timeout,
                disable_tls_verify=disable_tls_verify,
                attempt=attempt,
            )
            return (job.scene, True, None)
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip()
            if attempt > max_retries:
                return (job.scene, False, message)
            time.sleep(backoff)
            backoff *= 2

    return (job.scene, False, "exceeded retries")


def main() -> None:
    args = parse_args()
    direction = args.direction.upper()

    patch_dns()

    existing_root = Path(args.existing_root)
    target_root = Path(args.target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    username, password = get_earthdata_credentials()
    session = build_asf_session(username, password)

    summary: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    jobs_per_orbit: Dict[int, List[DownloadJob]] = {}

    remote_products = collect_remote_scenes(
        session=session,
        orbits=args.orbits,
        wkt=args.wkt,
        start_date=args.start_date,
        end_date=args.end_date,
        direction=direction,
    )

    for orbit in args.orbits:
        remote_for_orbit = remote_products.get(orbit, {})

        local_existing = collect_local_scenes(existing_root, orbit, direction)
        local_downloads = collect_local_scenes(target_root, orbit, direction)
        local_all = local_existing | local_downloads

        missing_scenes = sorted(set(remote_for_orbit.keys()) - local_all)
        summary[orbit]["remote"] = len(remote_for_orbit)
        summary[orbit]["local"] = len(local_all)
        summary[orbit]["missing"] = len(missing_scenes)

        if not missing_scenes:
            continue

        jobs_per_orbit[orbit] = prepare_download_jobs(
            orbit=orbit,
            scenes=missing_scenes,
            direction=direction,
            target_root=target_root,
        )

    if args.dry_run:
        print("Dry run mode - no downloads will be performed.")
        for orbit in args.orbits:
            stats = summary.get(orbit, {})
            print(
                f"[{direction} {orbit}] remote={stats.get('remote', 0)} "
                f"local={stats.get('local', 0)} missing={stats.get('missing', 0)}"
            )
            for job in jobs_per_orbit.get(orbit, []):
                print(f"  - {job.scene}")
        return

    if not jobs_per_orbit:
        for orbit in args.orbits:
            stats = summary.get(orbit, {})
            print(
                f"[{direction} {orbit}] up-to-date (remote={stats.get('remote', 0)}, "
                f"local={stats.get('local', 0)})"
            )
        return

    # Flatten job list for concurrent execution.
    all_jobs: List[DownloadJob] = []
    for orbit_jobs in jobs_per_orbit.values():
        all_jobs.extend(orbit_jobs)

    print(
        f"Starting downloads for {len(all_jobs)} scene(s) across "
        f"{len(jobs_per_orbit)} orbit(s) with {args.threads} worker(s)."
    )

    failed: List[Tuple[DownloadJob, Optional[str]]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                download_job,
                job,
                username,
                password,
                args.max_retries,
                args.session_timeout,
                args.no_verify,
                args.backoff_seconds,
            ): job
            for job in all_jobs
        }

        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            scene, success, message = future.result()
            if success:
                print(f"[OK] {direction} {scene}")
            else:
                print(f"[FAIL] {direction} {scene} :: {message}")
                failed.append((job, message))

    if failed:
        print("\nSummary: some downloads failed. Rerun the script to retry.")
        for job, message in failed:
            print(
                f"  - {direction} orbit {job.orbit} :: {job.scene} :: "
                f"{message or 'unknown error'}"
            )
        sys.exit(1)

    print("\nAll requested scenes downloaded successfully.")


if __name__ == "__main__":
    main()
