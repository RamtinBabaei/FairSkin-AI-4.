from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm

CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"


def download_file(url: str, dest: Path, timeout: int = 30, retries: int = 3) -> bool:
    headers = {"User-Agent": "FairSkin-AI research client/1.0"}
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return True
        except Exception:
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    return False


def main():
    p = argparse.ArgumentParser(description="Prepare Fitzpatrick17k metadata and optionally download source images.")
    p.add_argument("--out", default="data/fitzpatrick17k")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="0 means all rows")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_csv = out / "fitzpatrick17k.csv"
    if not raw_csv.exists():
        print("Downloading Fitzpatrick17k metadata...")
        if not download_file(CSV_URL, raw_csv):
            raise RuntimeError("Could not download Fitzpatrick17k metadata. Download the CSV manually from the official GitHub repository.")

    df = pd.read_csv(raw_csv)
    needed = {"md5hash", "fitzpatrick", "three_partition_label", "url"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected Fitzpatrick17k schema; missing columns: {sorted(missing)}")
    df = df[df["fitzpatrick"].isin([1,2,3,4,5,6])].copy()
    df = df[df["three_partition_label"].notna()].copy()
    if args.limit > 0:
        df = df.head(args.limit).copy()

    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    df["image_path"] = df["md5hash"].astype(str) + ".jpg"
    std = df[["image_path", "three_partition_label", "fitzpatrick", "url", "md5hash"]].rename(columns={"three_partition_label": "target"})

    if not args.metadata_only:
        failures = []
        jobs = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            for row in std.itertuples(index=False):
                dest = images / row.image_path
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                jobs.append((row.image_path, ex.submit(download_file, str(row.url), dest)))
            for name, future in tqdm(jobs, desc="Downloading images"):
                if not future.result():
                    failures.append(name)
        if failures:
            (out / "failed_downloads.txt").write_text("\n".join(failures), encoding="utf-8")
            print(f"Warning: {len(failures)} source images could not be downloaded. See failed_downloads.txt")
        std = std[std["image_path"].map(lambda x: (images / x).exists())].copy()

    std.to_csv(out / "metadata_standardized.csv", index=False)
    print(f"Prepared {len(std)} usable rows: {out / 'metadata_standardized.csv'}")
    print("Dataset use is subject to the original Fitzpatrick17k/image-source terms. Do not commit images to this repository.")


if __name__ == "__main__":
    main()
