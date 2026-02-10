from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "CP001": range(12),
        }
    )

    out_csv = outputs_dir / "hello_output.csv"
    df.to_csv(out_csv, index=False)

    print("BasinCast OK ✅")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()