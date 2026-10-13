"""
Mean-over-calls QoS of every closed-loop batch tested so far, with a relative
composite score per table.

Score: per call, every metric is min-max normalised across the algorithms of
the same table (1 = best of the table on that call): throughput (higher is
better), mean one-way delay, freeze ratio, packet loss (lower is better).
score3 = mean of throughput / delay / freeze, score4 adds packet loss; both
are then averaged over calls. Scores are only comparable inside one table.

usage: python summarize_all.py [out.txt]   (default localtest/all_algorithms_summary.txt;
       the per-call rows go to localtest/all_algorithms_percall.csv; both are local only)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"gcc": "GCC", "sjtu-metaband": "MetaBand(paper)", "miql_3c27_79200": "MetaBand(x3-s1)",
         "Schaferct_model": "Schaferct"}

# table -> [(tag, policy filter or None, suffix)], calls restricted to the first tag's calls
TABLES = [
    ("Env A (first): external audio flow, frame dropper off, capacity x1.0 - 148 calls",
     [("full148", None, "")]),
    ("Env B (calibrated): WebRTC Opus audio, frame dropper on, capacity x1.1 - 148 calls",
     [("full148cal", None, ""), ("full148cal_SG", None, "+SG")]),
    ("Env B pilot (16 calls): safety-layer variants of MetaBand(paper)",
     [("cal16", None, ""), ("cal16_S", None, "+S"), ("cal16_G", None, "+G"),
      ("cal16_SG", None, "+SG"), ("cal16_SG2", None, "+SG2")]),
    ("Env A pilot (16 calls): audio-allocation / guard variants of MetaBand(paper)",
     [("pilotF1", None, "+F1"), ("pilotF12", None, "+F1+F2"),
      ("full148", {"GCC", "MetaBand(paper)"}, "")]),
]
METRICS = [("throughput_mbps", "thr"), ("utilisation", "util"), ("video_bitrate_mbps", "video"),
           ("owd_mean_ms", "owd"), ("owd_p95_ms", "owd95"), ("pkt_loss_pct", "loss"),
           ("e2e_delay_p50_ms", "e2e50"), ("freeze_ratio_pct", "freeze")]
SCORE = {"thr": +1, "owd": -1, "freeze": -1, "loss": -1}


def load(tag, keep, suffix):
    rows = []
    for q in sorted(glob.glob(os.path.join(ROOT, "results_local", tag, "*", "*", "qos.json"))):
        run = os.path.basename(os.path.dirname(q))
        pol = run.rsplit("_", 1)[0]
        if pol not in NAMES or (keep and NAMES[pol] not in keep):
            continue
        r = json.load(open(q))
        r.update(call=os.path.basename(os.path.dirname(os.path.dirname(q))), run=run,
                 algo=NAMES[pol] + suffix)
        rows.append(r)
    return pd.DataFrame(rows)


def norm(df, col, sign):
    g = df.groupby("call")[col]
    lo, hi = g.transform("min"), g.transform("max")
    s = (df[col] - lo) / (hi - lo).replace(0, np.nan)
    s = (s if sign > 0 else 1 - s).fillna(1.0)          # all algorithms equal on this call
    return s.where(df[col].notna(), 0.0)                 # missing metric counts as the worst


def table(parts):
    df = pd.concat([load(*p) for p in parts], ignore_index=True)
    df = df.sort_values("run").groupby(["call", "algo"], as_index=False).last()  # latest rerun
    calls = set(load(*parts[0]).call)
    df = df[df.call.isin(calls)].rename(columns=dict(METRICS))
    df = df[df.groupby("call").algo.transform("nunique") == df.algo.nunique()]  # complete calls
    for k, sgn in SCORE.items():
        df["s_" + k] = norm(df, k, sgn)
    df["score3"] = df[["s_thr", "s_owd", "s_freeze"]].mean(axis=1)
    df["score4"] = df[["s_thr", "s_owd", "s_freeze", "s_loss"]].mean(axis=1)
    cols = [c for _, c in METRICS] + ["s_" + k for k in SCORE] + ["score3", "score4"]
    t = df.groupby("algo")[cols].mean()
    t["e2e50"] = df.groupby("algo").e2e50.median()          # median of per-call medians
    t.insert(0, "calls", df.groupby("algo").call.nunique())
    t = t.sort_values("score3", ascending=False)
    t.insert(0, "rank", range(1, len(t) + 1))
    return t, df


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "localtest", "all_algorithms_summary.txt")
    pd.set_option("display.width", 250)
    pd.set_option("display.precision", 3)
    text, per_call = [], []
    for title, parts in TABLES:
        t, df = table(parts)
        text.append(f"### {title}\n{t.to_string()}\n")
        per_call.append(df.assign(table=title))
    text.append("thr/video Mbps, owd/owd95/e2e50 ms, loss/freeze %; e2e50 = median over calls, "
                "others mean over calls; score3 = thr+owd+freeze, score4 = + loss (relative, per table)")
    s = "\n".join(text)
    print(s)
    open(out, "w").write(s + "\n")
    keep = ["table", "call", "algo", "run"] + [c for _, c in METRICS] + ["score3", "score4"]
    pd.concat(per_call)[keep].to_csv(os.path.join(ROOT, "localtest", "all_algorithms_percall.csv"), index=False)


if __name__ == "__main__":
    main()
