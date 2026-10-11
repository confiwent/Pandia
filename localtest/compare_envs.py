"""
QoS of the same calls and policies under two closed-loop batches (e.g. the
first environment `full148` vs the calibrated one `cal16`), median over calls
per capacity bin.

usage: python compare_envs.py <tagA> <tagB> [calls.txt]
"""
import glob
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"gcc": "GCC", "sjtu-metaband": "MetaBand(paper)", "miql_3c27_79200": "MetaBand(x3-s1)",
         "Schaferct_model": "Schaferct"}
COLS = ["utilisation", "owd_mean_ms", "owd_p95_ms", "pkt_loss_pct", "e2e_delay_p50_ms",
        "e2e_delay_p95_ms", "freeze_ratio_pct", "fps_decoded", "video_bitrate_mbps"]


def load(tag, calls):
    rows = []
    for q in glob.glob(os.path.join(ROOT, "results_local", tag, "*", "*", "qos.json")):
        call = os.path.basename(os.path.dirname(os.path.dirname(q)))
        if calls and call not in calls:
            continue
        run = os.path.basename(os.path.dirname(q))
        pol = run.rsplit("_", 1)[0]
        if pol not in NAMES:
            continue
        r = json.load(open(q))
        r.update(call=call, policy=NAMES[pol], env=tag)
        rows.append(r)
    return pd.DataFrame(rows).groupby(["call", "policy", "env"], as_index=False).last()


def main():
    a, b = sys.argv[1], sys.argv[2]
    calls = None
    if len(sys.argv) > 3:
        calls = {l.strip() for l in open(sys.argv[3]) if l.strip()}
    df = pd.concat([load(a, calls), load(b, calls)], ignore_index=True)
    cap = df.groupby("call").capacity_mbps.first()
    df["bin"] = df.call.map(lambda c: "<0.3" if cap[c] < 0.3 else ("0.3-1" if cap[c] < 1 else ("1-3" if cap[c] < 3 else ">3")))
    pd.set_option("display.width", 250)
    pd.set_option("display.precision", 2)
    for bn, sub in df.groupby("bin"):
        print(f"== {bn} Mbps ({sub.call.nunique()} calls), median over calls")
        t = sub.groupby(["policy", "env"])[COLS].median()
        print(t.to_string(), "\n")


if __name__ == "__main__":
    main()
