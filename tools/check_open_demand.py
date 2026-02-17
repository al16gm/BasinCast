from pathlib import Path
import pandas as pd
import numpy as np

CACHE = Path("outputs/cache/demand")
print("Cache dir:", CACHE.resolve())
print()

# 1) Find cached total series
cand = sorted(CACHE.glob("total_withdrawals_sectors_hm3_monthly_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not cand:
    print("No cached total demand series found yet.")
else:
    p = cand[0]
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    v = pd.to_numeric(df[df.columns[-1]], errors="coerce").fillna(0.0)
    print("CACHED TOTAL SERIES:", p.name)
    print("Rows:", len(df))
    print("Non-zero months:", int((v > 0).sum()), "/", len(v))
    print("Min/Mean/Max:", float(v.min()), float(v.mean()), float(v.max()))
    print("Head:\n", df.head(3))
    print("Tail:\n", df.tail(3))
    print()

# 2) Check canonical lat/lon sanity (if present)
canon = Path("outputs/_tmp_canonical_for_core.csv")
if canon.exists():
    cdf = pd.read_csv(canon)
    print("Canonical tmp file found:", canon)
    for col in ["lat", "lon"]:
        if col in cdf.columns:
            s = pd.to_numeric(cdf[col], errors="coerce")
            print(f"{col} range:", float(s.min()), "to", float(s.max()), "| NaN:", int(s.isna().sum()))
    print()

# 3) Inspect extracted sector CSVs (if extracted exists)
ex_dirs = [d for d in CACHE.glob("_extracted_*") if d.is_dir()]
if not ex_dirs:
    print("No extracted folder found yet (may be cleaned or not extracted).")
    raise SystemExit(0)

ex_dir = sorted(ex_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
print("Using extracted dir:", ex_dir)

csvs = sorted(ex_dir.glob("twd*_km3permonth.csv"))
print("Sector CSVs:", [c.name for c in csvs])

if not csvs:
    print("No twd*_km3permonth.csv found in extracted dir.")
    raise SystemExit(0)

# Read header & sample lat/lon stats to infer lon convention
sample = pd.read_csv(csvs[0], nrows=2000)
cols = list(sample.columns)
print("\nSample columns (first 20):", cols[:20])

# detect lat/lon columns
def detect_lat_lon(cols):
    low = {c.lower(): c for c in cols}
    lat = low.get("lat") or low.get("latitude") or low.get("y")
    lon = low.get("lon") or low.get("longitude") or low.get("x")
    return lat, lon

latc, lonc = detect_lat_lon(cols)
print("Detected lat col:", latc, "| lon col:", lonc)

if lonc:
    lonv = pd.to_numeric(sample[lonc], errors="coerce")
    print("Lon sample min/max:", float(lonv.min()), float(lonv.max()))
    if float(lonv.max()) > 180 and float(lonv.min()) >= 0:
        print(">>> Dataset lon convention likely 0..360 (IMPORTANT).")
    else:
        print(">>> Dataset lon convention likely -180..180.")
else:
    print("No lon column detected in sample; file may be in long or wide format without explicit lon.")

print("\nDONE.")
