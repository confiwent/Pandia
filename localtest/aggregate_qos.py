"""
Aggregate the QoS of a closed-loop batch (results_local/<tag>/<trace>/<policy>_<time>/qos.json).

Per policy: mean over calls (and the median for the tail metrics), overall and
per scenario of the test set: loss / no loss, constant / varying capacity,
behaviour-policy group v0-v4 (from subset148.txt). Only calls that finished
for every policy are used, so all policies are compared on the same calls.

usage: python aggregate_qos.py <tag> [--csv out.csv]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = [("throughput_mbps", "mean"), ("utilisation", "mean"), ("owd_mean_ms", "mean"),
           ("owd_p95_ms", "mean"), ("pkt_jitter_ms", "mean"), ("pkt_loss_pct", "mean"),
           ("e2e_delay_p50_ms", "median"), ("e2e_delay_p95_ms", "median"),
           ("freeze_count", "mean"), ("freeze_ratio_pct", "mean"), ("fps_decoded", "mean"),
           ("video_bitrate_mbps", "mean"), ("res_median_p", "mean")]
POLICY_NAME = {"gcc": "GCC", "miql_3c27_79200.onnx": "MetaBand(x3-s1)",
               "sjtu-metaband.onnx": "MetaBand(paper)", "Schaferct_model.onnx": "Schaferct"}


def trace_info():
    info = {}
    with open(os.path.join(ROOT, "docker_mnt/traffic_shell/subset148.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            group, name = line.split("/")
            d = json.load(open(os.path.join(ROOT, "docker_mnt/traffic_shell/trace_data", name)))
            c = np.asarray(d["true_capacity"], float)
            l = np.asarray(d["true_loss_rate"], float)
            info[os.path.splitext(name)[0]] = {
                "group": group, "capacity": "varying" if np.nanstd(c) > 0 else "constant",
                "loss": "loss" if np.nanmax(l) > 0 else "no loss",
                "mean_capacity_mbps": float(np.nanmean(c) / 1e6)}
    return info


def load(tag):
    rows = []
    for q in glob.glob(os.path.join(ROOT, "results_local", tag, "*", "*", "qos.json")):
        r = json.load(open(q))
        r["call"] = os.path.basename(os.path.dirname(os.path.dirname(q)))
        r["policy"] = POLICY_NAME.get(r["policy"], r["policy"])
        rows.append(r)
    df = pd.DataFrame(rows)
    # if a call/policy was run more than once keep the last run
    df = df.groupby(["call", "policy"], as_index=False).last()
    return df


def table(df):
    agg = {m: how for m, how in METRICS}
    t = df.groupby("policy").agg(agg)
    t.insert(0, "calls", df.groupby("policy").size())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--csv")
    a = ap.parse_args()
    info = trace_info()
    df = load(a.tag)
    n_pol = df.policy.nunique()
    complete = df.groupby("call").policy.nunique()
    df = df[df.call.isin(complete[complete == n_pol].index)]
    for k in ("group", "capacity", "loss", "mean_capacity_mbps"):
        df[k] = df.call.map(lambda c: info.get(c, {}).get(k))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.precision", 3)
    print(f"{df.call.nunique()} calls with all {n_pol} policies\n")
    print("== all calls"); print(table(df).to_string(), "\n")
    for key in ("loss", "capacity", "group"):
        for val, sub in df.groupby(key):
            print(f"== {key} = {val} ({sub.call.nunique()} calls)")
            print(table(sub).to_string(), "\n")
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
