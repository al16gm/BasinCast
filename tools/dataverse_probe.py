import json
from pathlib import Path

import requests

DOI = "doi:10.7910/DVN/VIQEAB"
URL = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={DOI}"

print("Requesting:", URL)
r = requests.get(URL, timeout=30)
print("Status:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type", ""))
print("First 200 chars:\n", (r.text or "")[:200])

# Save raw response (for reproducibility)
out_dir = Path("outputs/cache/demand")
out_dir.mkdir(parents=True, exist_ok=True)
raw_path = out_dir / "dataverse_dataset_raw.json"
raw_path.write_text(r.text or "", encoding="utf-8")
print("Saved raw JSON to:", raw_path)

# Parse JSON safely
try:
    data = r.json()
except Exception as e:
    print("ERROR: response is not JSON:", type(e).__name__, e)
    raise SystemExit(1)

print("Top-level keys:", list(data.keys())[:20])
print("API status field:", data.get("status"))

# Try to extract files list
files = []
try:
    files = data["data"]["latestVersion"]["files"]
except Exception:
    # fallback (some Dataverse responses differ)
    try:
        files = data["data"]["latestVersion"]["files"]
    except Exception as e:
        print("ERROR: cannot find file list in JSON:", type(e).__name__, e)
        raise SystemExit(1)

print("Files found:", len(files))

# Print a compact table of candidates
cands = []
for f in files:
    df = f.get("dataFile", {}) or {}
    file_id = df.get("id", None)
    label = df.get("filename", None) or f.get("label", None)
    ctype = df.get("contentType", None) or f.get("dataFile", {}).get("contentType", None)
    size = df.get("filesize", None)

    label_s = str(label or "")
    if any(k in label_s.lower() for k in ["monthly", "sector", "sectors", "withdraw", "demand", "total", "water"]):
        cands.append((file_id, label_s, ctype, size))

# Show first 30 candidates
print("\n--- Candidate files (id | label | contentType | size) ---")
for i, (fid, lab, ctype, size) in enumerate(cands[:30], start=1):
    print(f"{i:02d}. {fid} | {lab} | {ctype} | {size}")

# Suggest the “best” guess for monthly sectors zip/csv
best = None
for fid, lab, ctype, size in cands:
    L = lab.lower()
    if ("monthly" in L) and ("sector" in L or "sectors" in L) and (L.endswith(".zip") or L.endswith(".csv")):
        best = (fid, lab, ctype, size)
        break

print("\nSuggested best candidate:", best)
