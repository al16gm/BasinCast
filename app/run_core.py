from __future__ import annotations

import argparse
from pathlib import Path

from basincast.core.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BasinCast Core v0.5.1 (Paper-2 holdout protocol).")
    p.add_argument("--input", type=str, required=True, help="Path to canonical_timeseries.csv")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    p.add_argument("--horizons", type=str, default="12,24,36,48", help="Comma-separated horizons (months)")
    p.add_argument("--test-months", type=int, default=96, help="Holdout length in months (default: 96)")
    p.add_argument("--val-months", type=int, default=36, help="Inner validation months (default: 36)")
    p.add_argument("--kge-threshold", type=float, default=0.6, help="Reliability KGE threshold (default: 0.6)")
    p.add_argument("--min-history-months", type=int, default=120, help="Minimum history months per point")
    p.add_argument("--min-pairs", type=int, default=24, help="Minimum scored pairs per horizon")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())

    cfg = PipelineConfig(
        horizons=horizons,
        test_months=int(args.test_months),
        val_months=int(args.val_months),
        kge_threshold=float(args.kge_threshold),
        min_history_months=int(args.min_history_months),
        min_pairs_per_horizon=int(args.min_pairs),
    )

    forecasts_df, metrics_df, skill_df = run_pipeline(
        canonical_csv=Path(args.input),
        outdir=Path(args.outdir),
        cfg=cfg,
    )

    print("BasinCast Core v0.5.1 ✅")
    print(f"Forecasts rows: {len(forecasts_df)}")
    print(f"Metrics rows:   {len(metrics_df)}")
    print(f"Skill rows:     {len(skill_df)}")
    print(f"Saved: {Path(args.outdir) / 'forecasts_v0_5_1.csv'}")
    print(f"Saved: {Path(args.outdir) / 'metrics_v0_5_1.csv'}")
    print(f"Saved: {Path(args.outdir) / 'skill_by_horizon_v0_5_1.csv'}")


if __name__ == "__main__":
    main()