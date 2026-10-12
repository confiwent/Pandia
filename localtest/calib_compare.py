"""
Environment calibration: closed-loop replay of the test-set behaviour policy
(run_policy.py --policy replay) vs the observations recorded for the same call.

If the emulator reproduced Microsoft's emulated test environment, replaying
the behaviour policy's bandwidth estimates step by step would reproduce the
recorded observations. For every call and feature (newest short MI, newest
long MI) it reports the online / recorded medians and p90s, and the Pearson
correlation of the long-MI receiving rate and of the queuing delay over time
(steps aligned by index, start-up skipped).

usage: python calib_compare.py <tag>   (results_local/<tag>/<call>/replay_*/)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from qos_metrics import open_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = {"recv_rate_kbps": (5, 1e-3), "n_pkts_60ms": (10, 1), "bytes_per_pkt": (None, 1),
        "queue_ms": (30, 1), "delay_ms": (40, 1), "interarrival_ms": (80, 1), "jitter_ms": (90, 1),
        "loss_ratio": (100, 1), "video_share": (120, 1), "audio_share": (130, 1)}


def feats(o):
    out = {}
    for k, (i, sc) in FEAT.items():
        if k == "bytes_per_pkt":
            out[k] = o[:, 20] / np.maximum(o[:, 10], 1)
        else:
            out[k] = o[:, i] * sc
    return out


def main(tag):
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results_local", tag, "*", "replay_*"))):
        call = os.path.basename(os.path.dirname(d))
        rec = json.load(open_text(os.path.join(d, f"{call}.json")))
        on = np.asarray(rec["observations_rx"], float)
        ref = np.asarray(json.load(open(os.path.join(
            ROOT, "docker_mnt/traffic_shell/trace_data", f"{call}.json")))["observations"], float)
        cap = np.asarray(rec["true_capacity"], float)
        n = min(len(on), len(ref))
        s = slice(min(80, n // 10), n)
        fo, fr = feats(on[s]), feats(ref[s])
        r = {"call": call, "cap_mbps": float(np.nanmean(cap) / 1e6)}
        for k in FEAT:
            r[k + "_on"] = float(np.median(fo[k]))
            r[k + "_ref"] = float(np.median(fr[k]))
            r[k + "_p90_on"] = float(np.percentile(fo[k], 90))
            r[k + "_p90_ref"] = float(np.percentile(fr[k], 90))
        for k in ("recv_rate_kbps", "queue_ms"):
            a, b = fo[k], fr[k]
            r["corr_" + k] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df.cap_mbps, [0, 0.3, 1, 3, 100], labels=["<0.3", "0.3-1", "1-3", ">3"])
    pd.set_option("display.width", 250)
    pd.set_option("display.precision", 3)
    print(f"{len(df)} calls\n")
    summ = []
    for b, sub in df.groupby("bin", observed=True):
        for k in FEAT:
            summ.append({"bin": b, "feature": k, "online_med": sub[k + "_on"].median(),
                         "test_med": sub[k + "_ref"].median(),
                         "online_p90": sub[k + "_p90_on"].median(), "test_p90": sub[k + "_p90_ref"].median()})
    print(pd.DataFrame(summ).pivot(index="feature", columns="bin").to_string(), "\n")
    print("time-series correlation (median over calls):",
          df.groupby("bin", observed=True)[["corr_recv_rate_kbps", "corr_queue_ms"]].median().to_string())
    df.to_csv(os.path.join(ROOT, "localtest", f"calib_{tag}.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1])
