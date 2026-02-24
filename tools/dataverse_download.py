from pathlib import Path
import sys
import requests

if len(sys.argv) < 2:
    print("Usage: python tools\\dataverse_download.py <DATAFILE_ID>")
    raise SystemExit(1)

file_id = sys.argv[1].strip()
url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"

out_dir = Path("outputs/cache/demand")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / f"dataverse_file_{file_id}.bin"

print("Downloading:", url)
with requests.get(url, stream=True, timeout=120) as r:
    print("Status:", r.status_code)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)

print("Saved to:", out_path)
