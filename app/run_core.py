from __future__ import annotations

import argparse
from pathlib import Path

from basincast.core.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BasinCast Core v0.5 (ENDO-only) runner.")
    p.add_argument("--input", type=str, required=True, help="Path to canonical_timeseries.csv")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--horizons", type=str, default="1,3,6,12,24,36,48", help="Comma-separated horizons in months")
    p.add_argument("--val-months", type=int, default=36, help="Validation months (last N months)")
    p.add_argument("--min-history-months", type=int, default=120, help="Minimum monthly history required per point")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    cfg = PipelineConfig(
        horizons=horizons,
        val_months=int(args.val_months),
        min_history_months=int(args.min_history_months),
    )

    forecasts_df, metrics_df = run_pipeline(
        canonical_csv=Path(args.input),
        outdir=Path(args.outdir),
        cfg=cfg,
    )

    print("BasinCast Core v0.5 ✅")
    print(f"Forecasts rows: {len(forecasts_df)}")
    print(f"Metrics rows:   {len(metrics_df)}")
    print(f"Saved: {Path(args.outdir) / 'forecasts_v0_5.csv'}")
    print(f"Saved: {Path(args.outdir) / 'metrics_v0_5.csv'}")


if __name__ == "__main__":
    main()