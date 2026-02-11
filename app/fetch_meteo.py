from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from basincast.exog.power import fetch_power_monthly_point


def _parse_dates_monthly(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
    if df[col].isna().any():
        bad = df[df[col].isna()].head(5)
        raise ValueError(f"Some dates could not be parsed. Examples:\n{bad}")
    # force first day of month
    df[col] = df[col].dt.to_period("M").dt.to_timestamp()
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            # handle comma decimals
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch monthly meteo (NASA POWER) and merge into canonical.")
    ap.add_argument("--input", required=True, help="Path to canonical_timeseries.csv")
    ap.add_argument("--out_canonical", default="outputs/canonical_with_meteo.csv", help="Merged output path")
    ap.add_argument("--out_meteo", default="outputs/meteo_power_monthly.csv", help="Meteo-only output path")
    ap.add_argument("--community", default="AG", choices=["AG", "RE", "SB"], help="NASA POWER community")
    ap.add_argument("--cache_dir", default=".cache/power", help="Cache directory")
    ap.add_argument("--sleep_s", type=float, default=0.1, help="Polite sleep between points")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    df = pd.read_csv(inp)
    required = {"point_id", "date", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in canonical: {sorted(missing)}")

    df = _parse_dates_monthly(df, "date")
    df = _coerce_numeric(df, ["lat", "lon", "value"])

    # Determine year-range from the canonical itself
    start_year = int(df["date"].min().year)
    end_year = int(df["date"].max().year)

    meteo_all = []
    points = (
        df[["point_id", "lat", "lon"]]
        .dropna()
        .drop_duplicates()
        .sort_values("point_id")
        .to_dict("records")
    )

    if len(points) == 0:
        raise ValueError("No valid point_id/lat/lon rows found.")

    print(f"Points found: {len(points)} | Years: {start_year}-{end_year} | Community: {args.community}")

    for i, p in enumerate(points, 1):
        pid = p["point_id"]
        lat = float(p["lat"])
        lon = float(p["lon"])

        print(f"[{i}/{len(points)}] Fetching NASA POWER monthly for point_id={pid} (lat={lat}, lon={lon}) ...")
        df_m = fetch_power_monthly_point(
            lat=lat,
            lon=lon,
            start_year=start_year,
            end_year=end_year,
            community=args.community,
            cache_dir=args.cache_dir,
            sleep_s=args.sleep_s,
        )
        df_m["point_id"] = pid
        df_m["lat"] = lat
        df_m["lon"] = lon
        meteo_all.append(df_m)

    meteo = pd.concat(meteo_all, ignore_index=True)

    # Merge into canonical by (point_id, date)
    merged = df.merge(
        meteo[["point_id", "date", "precip_mm_day", "precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c", "source"]],
        on=["point_id", "date"],
        how="left",
    )

    out_c = Path(args.out_canonical)
    out_m = Path(args.out_meteo)
    out_c.parent.mkdir(parents=True, exist_ok=True)
    out_m.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_c, index=False)
    meteo.to_csv(out_m, index=False)

    print("✅ Done")
    print(f"Saved merged canonical: {out_c}")
    print(f"Saved meteo-only:       {out_m}")


if __name__ == "__main__":
    main()