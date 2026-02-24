# tools/download_tethys_demand.py
from __future__ import annotations

import argparse
from pathlib import Path

import requests


DATAVERSE_ACCESS_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"


def download(file_id: int, dest: Path, verify: bool = True, timeout: int = 120, chunk_mb: int = 8) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[OK] Already exists: {dest} ({dest.stat().st_size} bytes)")
        return

    url = DATAVERSE_ACCESS_URL.format(file_id=file_id)
    params = {"format": "original"}

    tmp = dest.with_suffix(dest.suffix + ".part")
    chunk_bytes = chunk_mb * 1024 * 1024

    print(f"[DL] {url} -> {dest}")
    with requests.get(url, params=params, stream=True, timeout=timeout, verify=verify) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_bytes):
                if chunk:
                    f.write(chunk)

    tmp.replace(dest)
    print(f"[OK] Downloaded: {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=r"outputs\cache\demand")
    ap.add_argument("--verify", action="store_true", default=True)
    ap.add_argument("--no-verify", action="store_true", default=False)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    verify = args.verify and (not args.no_verify)

    cache = Path(args.cache)
    # Your current chosen pair (ssp1_rcp26, gfdl, withdrawals sectors monthly)
    download(6062173, cache / "ssp1_rcp26_gfdl_withdrawals_sectors_monthly_1.zip", verify=verify, timeout=args.timeout)
    download(6062170, cache / "ssp1_rcp26_gfdl_withdrawals_sectors_monthly_2.zip", verify=verify, timeout=args.timeout)


if __name__ == "__main__":
    main()