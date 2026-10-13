"""
Which emulator settings reproduce the network the test-set calls saw?

For every batch tag (results_local/<tag>/<call>/<policy>_*/) and policy
(replay = the test set's own behaviour estimates, gcc = WebRTC GoogCC) it
compares the receiver-side observations of the run with the recorded
observations of the same call, per capacity bin (median over calls):
queuing delay p50 / p90 of the newest short MI, loss ratio, lost packets per
loss event (MIs with loss) and the long-MI receiving rate.

usage: python calib_env_compare.py <tag> [<tag> ...]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from qos_metrics import open_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.join(ROOT, "docker_mnt/traffic_shell/trace_data")


def stats(o):
    o = o[min(80, len(o) // 10):]
    m = o[:, 100] > 0
    return {"q50": float(np.median(o[:, 30])), "q90": float(np.percentile(o[:, 30], 90)),
            "loss": float(o[:, 100].mean()), "lost_ev": float(o[m, 110].mean()) if m.sum() >= 10 else np.nan,
            "rr": float(o[:, 5].mean() / 1e3)}


def main(tags):
    rows, ref_cache = [], {}
    for tag in tags:
        for d in sorted(glob.glob(os.path.join(ROOT, "results_local", tag, "*", "*_*"))):
            call = os.path.basename(os.path.dirname(d))
            pol = os.path.basename(d).rsplit("_", 1)[0]
            if pol not in ("replay", "gcc"):
                continue
            try:
                rec = json.load(open_text(os.path.join(d, f"{call}.json")))
            except (FileNotFoundError, OSError):
                continue
            if call not in ref_cache:
                tr = json.load(open(os.path.join(TRACES, f"{call}.json")))
                cap = np.nanmean(np.asarray([np.nan if v is None else v for v in tr["true_capacity"]], float))
                tl = np.nanmean(np.asarray([np.nan if v is None else v for v in tr["true_loss_rate"]], float))
                ref_cache[call] = (stats(np.asarray(tr["observations"], float)), cap / 1e6, tl)
            ref, cap, tl = ref_cache[call]
            on = stats(np.asarray(rec["observations_rx"], float))
            rows.append({"tag": tag, "policy": pol, "call": call, "cap": cap, "lossy": tl > 0,
                         **{k + "_on": v for k, v in on.items()}, **{k + "_ref": v for k, v in ref.items()}})
    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df.cap, [0, 0.3, 1, 3, 100], labels=["<0.3", "0.3-1", "1-3", ">3"])
    pd.set_option("display.width", 250)
    pd.set_option("display.precision", 2)
    for pol, g in df.groupby("policy"):
        print(f"=== {pol} (median over calls; *_ref = recorded test-set call)")
        cols = ["q50_on", "q50_ref", "q90_on", "q90_ref", "loss_on", "loss_ref", "rr_on", "rr_ref"]
        t = g.groupby(["bin", "tag"], observed=True)[cols].median()
        t.insert(0, "calls", g.groupby(["bin", "tag"], observed=True).call.nunique())
        print(t.to_string(), "\n")
        lossy = g[g.lossy]
        if len(lossy):
            print(f"  lossy calls ({lossy.call.nunique()}): lost packets per loss event, mean over calls")
            print(lossy.groupby("tag")[["lost_ev_on", "lost_ev_ref", "loss_on", "loss_ref"]].mean().to_string(), "\n")
    df.to_csv(os.path.join(ROOT, "localtest", "stats_local", "calib_env_compare.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
