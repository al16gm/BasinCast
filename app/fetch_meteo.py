from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from basincast.meteo.power import PowerMonthlyConfig, fetch_power_monthly
from basincast.meteo.geocode import GeocodeConfig, reverse_geocode_nominatim


def to_month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()


def main() -> None:
    ap = argparse.ArgumentParser(description="BasinCast v0.6.2 - Fetch monthly meteo (no API key) and merge into canonical.")
    ap.add_argument("--input", required=True, help="canonical_timeseries.csv with at least: point_id,date,lat,lon")
    ap.add_argument("--outdir", default="outputs", help="Output folder (default: outputs)")
    ap.add_argument("--with_geocode", action="store_true", help="Add city/state/country via Nominatim reverse geocoding (polite, cached).")
    ap.add_argument("--user_agent", default="", help="Required if --with_geocode. Example: BasinCast/0.6.2 (contact: you@domain.com)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column.")
    df["date"] = to_month_start(df["date"])

    required = {"point_id", "date", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for meteo fetch: {sorted(missing)}")

    # build meteo per point_id
    meteo_all = []
    geo_all = []

    for pid, g in df.groupby("point_id", sort=True):
        # take most common coordinates for the point
        lat = float(g["lat"].dropna().mode().iloc[0])
        lon = float(g["lon"].dropna().mode().iloc[0])

        start_year = int(pd.to_datetime(g["date"].min()).year)
        end_year = int(pd.to_datetime(g["date"].max()).year)

        met = fetch_power_monthly(lat=lat, lon=lon, start_year=start_year, end_year=end_year, cfg=PowerMonthlyConfig())
        met["point_id"] = pid
        meteo_all.append(met)

        if args.with_geocode:
            if not args.user_agent.strip():
                raise ValueError("If --with_geocode is set, you must provide --user_agent.")
            geo = reverse_geocode_nominatim(lat=lat, lon=lon, cfg=GeocodeConfig(user_agent=args.user_agent.strip()))
            geo["point_id"] = pid
            geo["lat"] = lat
            geo["lon"] = lon
            geo_all.append(pd.DataFrame([geo]))

    meteo_df = pd.concat(meteo_all, ignore_index=True) if meteo_all else pd.DataFrame()
    meteo_path = outdir / "meteo_power_monthly.csv"
    meteo_df.to_csv(meteo_path, index=False)

    merged = df.merge(
        meteo_df[["point_id", "date", "precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c", "source"]],
        on=["point_id", "date"],
        how="left",
    )

    if args.with_geocode and geo_all:
        geo_df = pd.concat(geo_all, ignore_index=True)
        geo_path = outdir / "locations_reverse_geocoded.csv"
        geo_df.to_csv(geo_path, index=False)
        # attach just the admin fields per point
        merged = merged.merge(
            geo_df.drop(columns=["lat", "lon"]).drop_duplicates(subset=["point_id"]),
            on="point_id",
            how="left",
        )

    merged_path = outdir / "canonical_with_meteo.csv"
    merged.to_csv(merged_path, index=False)

    print("BasinCast v0.6.2 ✅ Meteo fetch complete")
    print(f"Saved: {meteo_path}")
    if args.with_geocode and geo_all:
        print(f"Saved: {outdir / 'locations_reverse_geocoded.csv'}")
    print(f"Saved: {merged_path}")
    print(f"Merged rows: {len(merged)} | Points: {merged['point_id'].nunique()}")


if __name__ == "__main__":
    main()