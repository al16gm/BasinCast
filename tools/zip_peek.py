from pathlib import Path
import zipfile

z1 = Path("outputs/cache/demand/ssp1_rcp26_gfdl_withdrawals_sectors_monthly_1.zip")
z2 = Path("outputs/cache/demand/ssp1_rcp26_gfdl_withdrawals_sectors_monthly_2.zip")

for z in [z1, z2]:
    print("\nZIP:", z, "exists:", z.exists(), "size(bytes):", z.stat().st_size if z.exists() else None)
    if not z.exists():
        continue
    with zipfile.ZipFile(z, "r") as Z:
        names = Z.namelist()
        print("Files inside:", len(names))
        # show first 30
        for n in names[:30]:
            print(" -", n)
        # show CSV count
        csvs = [n for n in names if n.lower().endswith(".csv")]
        print("CSV files:", len(csvs))
