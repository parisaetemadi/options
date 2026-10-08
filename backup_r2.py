"""
Backup copy of data/ to Cloudflare R2 (bucket options-data, free up to 10 GB/month). The local files stay:
the tests read them from the laptop. Only new or changed files are uploaded (compared by size and MD5).

Needs in .env: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (an R2 token limited to this bucket).

Run:  python backup_r2.py            (upload what's new, then check every file is there with the same MD5)
      python backup_r2.py --check    (only check)
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig

ROOT = Path(__file__).parent
BUCKET = "options-data"
LIMIT_GB = 9.5                       # stay under the free 10 GB
# one part per file up to 5 GB, so R2's ETag is the file's MD5 and the check can compare them
ONE_PART = TransferConfig(multipart_threshold=5 * 1024**3, max_concurrency=1)


def client():
    env = dict(l.split("=", 1) for l in (ROOT / ".env").read_text().splitlines() if "=" in l and not l.startswith("#"))
    return boto3.client("s3", endpoint_url=env["R2_ENDPOINT"], region_name="auto",
                        aws_access_key_id=env["R2_ACCESS_KEY_ID"], aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"])


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def remote(s3):
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        out |= {o["Key"]: (o["Size"], o["ETag"].strip('"')) for o in page.get("Contents", [])}
    return out


def main(check_only):
    s3 = client()
    files = [p for p in sorted((ROOT / "data").rglob("*")) if p.is_file() and p.name not in (".lock", ".DS_Store")]
    total = sum(p.stat().st_size for p in files) / 1024**3
    print(f"{len(files)} local files, {total:.2f} GB")
    if total > LIMIT_GB:
        raise SystemExit(f"{total:.2f} GB is over the {LIMIT_GB} GB free limit -- not uploading")
    have = remote(s3)
    local = {p.relative_to(ROOT).as_posix(): p for p in files}
    with ThreadPoolExecutor(8) as pool:
        sums = dict(zip(local, pool.map(md5, local.values())))
    todo = [k for k, p in local.items() if have.get(k) != (p.stat().st_size, sums[k])]
    if not check_only and todo:
        print(f"Uploading {len(todo)} files ({sum(local[k].stat().st_size for k in todo) / 1024**3:.2f} GB)...")
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda k: s3.upload_file(str(local[k]), BUCKET, k, Config=ONE_PART), todo))
        have = remote(s3)
    bad = [k for k, p in local.items() if have.get(k) != (p.stat().st_size, sums[k])]
    size = sum(s for s, _ in have.values()) / 1024**3
    print(f"R2 holds {len(have)} files, {size:.2f} GB; {len(local) - len(bad)}/{len(local)} local files match (size + MD5)")
    if bad:
        print("Missing or different:", *bad[:20], sep="\n  ")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    main(p.parse_args().check)
