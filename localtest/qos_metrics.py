"""
Deployment (QoS) metrics of one closed-loop run of run_policy.py.

From the WebRTC log (pandia.log) of the run:
  * video frames: FrameCaptured (utc ts), OpenH264 Start/Finish encoding
    (resolution, size), SendPacket (rtp seq -> frame id, first packet of the
    frame), Frame decoding acked (first rtp seq, decoded utc ts)
  * end-to-end frame delay = decoded utc - captured utc (same host clock)
  * freezes as in WebRTC's getStats (RTCInboundRtpStreamStats.freezeCount /
    totalFreezesDuration): a decoded-frame interval is a freeze when it exceeds
    max(3 x avg, avg + 150 ms), avg = mean of the previous 30 intervals
From the receiver-side observations saved per step (<trace>.json):
  * throughput = receiving rate of the newest 60 ms MI (video + audio)
  * one-way packet delay = propagation delay + queuing delay of that MI
  * packet loss = sum lost / sum (lost + received) over the 60 ms MIs
  * jitter = packet interarrival jitter feature (std of gaps) of that MI

usage: python qos_metrics.py <result_dir> [...]   (prints one row per run, writes qos.json)
"""
import json
import os
import re
import sys

import numpy as np

RE = {
    "cap": re.compile(r"FrameCaptured, id: (\d+), width: \d+, height: (\d+), ts: \d+, utc ts: (\d+) ms"),
    "enc0": re.compile(r"OpenH264 Start encoding, frame id: (\d+), shape: (\d+) x (\d+), bitrate: (\d+) kbps"),
    "enc1": re.compile(r"Finish encoding, frame id: (\d+), frame type: \d+, frame size: (\d+), is key: (\d), qp: (\d+)"),
    "send": re.compile(r"SendPacket, id: \d+, seq: (\d+), first in frame: 1, last in frame: \d, fid: (\d+), type: 1,"),
    "dec": re.compile(r"Frame decoding acked, id: (\d+), receiving ts: (\d+), decoding ts: (\d+), decoded ts: (\d+)"),
}


def parse_log(path):
    cap, enc, seq2fid, dec = {}, {}, {}, []
    with open(path, errors="ignore") as f:
        for line in f:
            if "FrameCaptured" in line:
                m = RE["cap"].search(line)
                if m:
                    cap[int(m[1])] = int(m[3])
            elif "Start encoding" in line:
                m = RE["enc0"].search(line)
                if m:
                    enc.setdefault(int(m[1]), {})["height"] = int(m[3])
            elif "Finish encoding" in line:
                m = RE["enc1"].search(line)
                if m:
                    e = enc.setdefault(int(m[1]), {})
                    e["size"], e["qp"] = int(m[2]), int(m[4])
            elif "SendPacket" in line and "first in frame: 1" in line:
                m = RE["send"].search(line)
                if m:
                    seq2fid.setdefault(int(m[1]), int(m[2]))   # first send, not the RTX
            elif "Frame decoding acked" in line:
                m = RE["dec"].search(line)
                if m:
                    dec.append((int(m[1]), int(m[4])))
    return cap, enc, seq2fid, dec


def freezes(decoded_ms, window=30):
    """WebRTC-style freezes on the sequence of decoded-frame times (ms)."""
    iv = np.diff(np.asarray(decoded_ms, float))
    count, total = 0, 0.0
    for i in range(1, len(iv)):
        avg = iv[max(0, i - window):i].mean()
        if iv[i] > max(3 * avg, avg + 150):
            count += 1
            total += iv[i]
    return count, total


def qos(result_dir):
    rec_path = [os.path.join(result_dir, f) for f in os.listdir(result_dir)
                if f.endswith(".json") and f not in ("summary.json", "qos.json", "net_config.json")][0]
    rec = json.load(open(rec_path))
    cap, enc, seq2fid, dec = parse_log(os.path.join(result_dir, "pandia.log"))

    # video frames
    dec = sorted(set(dec), key=lambda x: x[1])
    dec_ms = np.array([d[1] for d in dec], float)
    duration_s = (dec_ms[-1] - dec_ms[0]) / 1000 if len(dec_ms) > 1 else np.nan
    e2e = np.array([d[1] - cap[seq2fid[d[0]]] for d in dec
                    if d[0] in seq2fid and seq2fid[d[0]] in cap], float)
    e2e = e2e[(e2e >= 0) & (e2e < 60000)]
    n_frz, frz_ms = freezes(dec_ms)
    enc_ok = [e for e in enc.values() if e.get("size", 0) > 0]
    enc_bits = sum(e["size"] for e in enc_ok) * 8
    heights = np.array([e["height"] for e in enc_ok if "height" in e])

    # network, from the receiver-side observation of every step (newest 60 ms MI)
    o = np.asarray(rec.get("observations_rx") or rec["observations"], float)
    cap_bps = np.asarray(rec["true_capacity"], float)
    skip = min(50, len(o) // 10)                      # start-up
    o, cap_bps = o[skip:], cap_bps[skip:]
    rate = o[:, 0]
    q = o[:, 30]
    n, lost = o[:, 10], o[:, 110]
    delay_ms = float(rec.get("delay_ms", 20))

    out = {
        "trace": rec.get("trace"), "policy": os.path.basename(str(rec.get("policy"))),
        "throughput_mbps": float(rate.mean() / 1e6),
        "capacity_mbps": float(np.nanmean(cap_bps) / 1e6),
        "utilisation": float(rate.mean() / np.nanmean(cap_bps)),
        "owd_mean_ms": float(delay_ms + q.mean()),
        "owd_p95_ms": float(delay_ms + np.percentile(q, 95)),
        "pkt_jitter_ms": float(o[:, 90].mean()),
        "pkt_loss_pct": float(100 * lost.sum() / max(1.0, (lost + n).sum())),
        "frames_captured": len(cap), "frames_decoded": len(dec),
        "fps_decoded": float(len(dec) / duration_s) if duration_s else np.nan,
        "e2e_delay_mean_ms": float(e2e.mean()) if len(e2e) else np.nan,
        "e2e_delay_p50_ms": float(np.median(e2e)) if len(e2e) else np.nan,
        "e2e_delay_p95_ms": float(np.percentile(e2e, 95)) if len(e2e) else np.nan,
        "e2e_delay_std_ms": float(e2e.std()) if len(e2e) else np.nan,
        "freeze_count": n_frz,
        "freeze_ratio_pct": float(100 * frz_ms / 1000 / duration_s) if duration_s else np.nan,
        "video_bitrate_mbps": float(enc_bits / duration_s / 1e6) if duration_s else np.nan,
        "res_median_p": float(np.median(heights)) if len(heights) else np.nan,
        "qp_mean": float(np.mean([e["qp"] for e in enc_ok if "qp" in e])) if enc_ok else np.nan,
    }
    with open(os.path.join(result_dir, "qos.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


COLS = [("policy", "{:>16s}"), ("throughput_mbps", "{:6.3f}"), ("utilisation", "{:5.2f}"),
        ("owd_mean_ms", "{:6.1f}"), ("owd_p95_ms", "{:6.1f}"), ("pkt_jitter_ms", "{:5.1f}"),
        ("pkt_loss_pct", "{:5.1f}"), ("e2e_delay_p50_ms", "{:6.0f}"), ("e2e_delay_p95_ms", "{:6.0f}"),
        ("freeze_count", "{:4d}"), ("freeze_ratio_pct", "{:5.1f}"), ("fps_decoded", "{:5.1f}"),
        ("video_bitrate_mbps", "{:5.3f}"), ("res_median_p", "{:5.0f}")]

if __name__ == "__main__":
    print(" ".join(c for c, _ in COLS))
    for d in sys.argv[1:]:
        r = qos(d)
        print(" ".join(fmt.format(r[c]) for c, fmt in COLS))
