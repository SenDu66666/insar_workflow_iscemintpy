#!/usr/bin/env bash
set -euo pipefail

# Sequential download helper for Canary Islands Sentinel-1 SLC data.
# Usage:
#   nohup bash repository/scripts/run_canarias_downloads.sh > /mnt/beegfs/procesados/sendu/canarias_download.log 2>&1 &

ENV_CMD="conda run -n mintpy2025 python scripts/download_missing_s1.py"
WKT_CANARY="POLYGON((-20.0 25.0, -20.0 30.0, -12.0 30.0, -12.0 25.0, -20.0 25.0))"
YEARS=("2022" "2023" "2024" "2025")

download_range() {
    local direction=$1
    shift
    local orbit_list=("$@")

    for year in "${YEARS[@]}"; do
        local start="${year}-01-01"
        local end="${year}-12-31"
        echo "=== ${direction} orbits ${orbit_list[*]} :: ${start} to ${end} ==="
        ${ENV_CMD} \
            --direction "${direction}" \
            --orbits "${orbit_list[@]}" \
            --start-date "${start}" \
            --end-date "${end}" \
            --wkt "${WKT_CANARY}" \
            --threads 3 \
            --session-timeout 600
    done
}

download_range ASCENDING 60 89 162
download_range DESCENDING 169 23 125

echo "All download jobs finished."
