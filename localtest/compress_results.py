"""
Losslessly gzip the bulky files of finished closed-loop runs:
results_local/<tag>/<call>/<run>/{pandia.log, <call>.json} -> *.gz
(qos.json, summary.json and the plots are left alone). A file is replaced only
after its .gz has been written and verified. Readers (qos_metrics.open_text)
accept both forms.

usage: python compress_results.py [results_local] [--workers N]
"""
import argparse
import glob
import gzip
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

SMALL = {"qos.json", "summary.json", "net_config.json"}


def targets(root):
    for run in glob.glob(os.path.join(root, "*", "*", "*")):
        if not os.path.isdir(run):
            continue
        for f in os.listdir(run):
            p = os.path.join(run, f)
            if f == "pandia.log" or (f.endswith(".json") and f not in SMALL):
                yield p


def compress(path):
    gz = path + ".gz"
    with open(path, "rb") as src, gzip.open(gz + ".tmp", "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    with gzip.open(gz + ".tmp", "rb") as chk:      # verify the archive reads back fully
        n = sum(len(b) for b in iter(lambda: chk.read(1 << 20), b""))
    if n != os.path.getsize(path):
        os.remove(gz + ".tmp")
        return path, 0, "size mismatch"
    saved = os.path.getsize(path) - os.path.getsize(gz + ".tmp")
    os.replace(gz + ".tmp", gz)
    os.remove(path)
    return path, saved, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_local"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = list(targets(a.root))
    print(f"{len(files)} files to compress", flush=True)
    saved, bad = 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (p, s, err) in enumerate(ex.map(compress, files, chunksize=8)):
            saved += s
            if err:
                bad += 1
                print("FAILED", p, err, flush=True)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(files)} saved {saved / 1e9:.1f} GB", flush=True)
    print(f"done: saved {saved / 1e9:.1f} GB, {bad} failed", flush=True)


if __name__ == "__main__":
    main()
