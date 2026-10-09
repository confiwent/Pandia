"""
Closed-loop test of an MMSys'24-style bandwidth estimator (or GCC) in the
Pandia emulator, with the trace replayed by emulator/trace_replay.py.

Policy interface (same as metaband_seg/eval_rtc_models.py):
  obs  = raw 150-dim observation * NORMAL_VECTOR, fed as `obs` [1, 1, 150];
         recurrent inputs are zeros. The observation is computed on the
         receiver side from captured RTP packets (rx_features.py, default) or
         taken from Pandia's sender-side array_bec() (--features sender);
         both are logged.
  out  = output[0, 0, 0], the bandwidth estimate in bps
The estimate drives the public Pandia WebRTC through its own shm fields:
  field 0 (encoder target bitrate) = estimate
  field 1 (pacing rate)            = PACING_FACTOR * estimate (WebRTC default 2.5)
`--policy gcc` writes 0 to both, i.e. WebRTC's GCC is in control.

Network: trace replay with an RTCP bypass and a bottleneck queue of --queue-ms
(default 1000 ms; per-call max queuing delay in the test set: median 240 ms,
p90 955 ms) (emulator/trace_replay.py) plus an Opus-like audio flow whose frame
interval follows the trace's audio packet rate (emulator/audio_flow.py).

Outputs in results_local/<trace>/<tag>_<time>/:
  <trace>.json  observations (the policy input) / bandwidth_predictions /
                true_capacity per step (MMSys'24 JSON layout, bps), plus
                observations_rx / observations_tx and timestamps
  summary.json  ER / OER / MSE (same formulas as the offline evaluation) and QoS
  Pandia's QoS plots (generate_diagrams), freeze.txt, score.txt, pandia.log

usage (inside pandia-driver:local, repo at /data2/kj/Workspace/Pandia):
    python localtest/run_policy.py 07482.json --policy localtest/models/miql_3c27_79200.onnx
    python localtest/run_policy.py 07482.json --policy gcc
"""
import argparse
import json
import os
import shutil
import time
import uuid
from datetime import datetime

import numpy as np
import onnxruntime as ort

import pandia.agent.env_emulator_offline as E
from pandia.agent.action import Action
from pandia.agent.env_config_offline import ENV_CONFIG
from pandia.analysis.stream_illustrator import generate_diagrams
from pandia.constants import K

import run_rbwe_local  # noqa: F401  (patches start_container for this host)
from rx_features import RxFeatures

PACING_FACTOR = 2.5
PROJECT = "/data2/kj/Workspace/Pandia"


def normal_vector():
    nv = np.empty(150, dtype=np.float32)
    for lo, hi, v in [(0, 10, 1e-6), (10, 15, 1e-1), (15, 20, 1e-2), (20, 25, 1e-4),
                      (25, 30, 1e-5), (30, 40, 1e-1), (40, 60, 1e-2), (60, 70, 1.0),
                      (70, 100, 1e-1), (100, 150, 1.0)]:
        nv[lo:hi] = v
    return nv


class OnnxPolicy:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.extras = {}
        for inp in self.sess.get_inputs():
            if inp.name != "obs":
                shape = [1 if (not isinstance(d, int) or d <= 0) else d for d in inp.shape]
                self.extras[inp.name] = np.zeros(shape, dtype=np.float32)
        self.nv = normal_vector()

    def __call__(self, obs_raw):
        feed = {"obs": (obs_raw * self.nv).astype(np.float32).reshape(1, 1, -1)}
        feed.update(self.extras)
        return float(self.sess.run(None, feed)[0][0, 0, 0])


def trace_audio_interval_ms(path):
    """Audio frame interval of a test-set call: median audio packets per 60 ms MI
    (packets x audio share of the newest short MI), snapped to 20/30/40/60 ms."""
    o = np.asarray(json.load(open(path))["observations"], dtype=float)
    per_mi = np.median(o[:, 10] * o[:, 130])
    if per_mi <= 0:
        return 20.0
    return float(min((20, 30, 40, 60), key=lambda iv: abs(60.0 / iv - per_mi)))


def metrics(pred_bps, true_bps):
    p, t = np.asarray(pred_bps, float) / 1e6, np.asarray(true_bps, float) / 1e6
    ok = ~(np.isnan(p) | np.isnan(t) | (t <= 0))
    p, t = p[ok], t[ok]
    return {"ER": float(np.mean(np.minimum(1, np.abs(p - t) / t))),
            "OER": float(np.mean(np.maximum(0, (p - t) / t))),
            "MSE": float(np.mean((p - t) ** 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--policy", required=True, help="path to .onnx, or 'gcc'")
    ap.add_argument("--delay", type=float, default=20, help="one-way delay, ms")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = whole trace")
    ap.add_argument("--min-bps", type=float, default=20e3)
    ap.add_argument("--features", choices=["receiver", "sender"], default="receiver")
    ap.add_argument("--queue-ms", type=int, default=1000, help="bottleneck queue (tbf latency)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--audio-interval-ms", type=float, default=0,
                    help="audio frame interval; 0 = from the trace's audio packet rate")
    a = ap.parse_args()
    if not a.audio_interval_ms:
        a.audio_interval_ms = trace_audio_interval_ms(
            f"{PROJECT}/docker_mnt/traffic_shell/trace_data/{a.trace}")

    pkt_sock = f"/tmp/{uuid.uuid4().hex[:8]}_pkt.sock"
    os.environ.update({"QUEUE_MS": str(a.queue_ms), "AUDIO": "0" if a.no_audio else "1",
                       "AUDIO_INTERVAL_MS": str(a.audio_interval_ms),
                       "PKT_SOCKET_PATH": pkt_sock})
    rx = RxFeatures(pkt_sock)

    is_gcc = a.policy == "gcc"
    policy = None if is_gcc else OnnxPolicy(a.policy)
    tag = "gcc" if is_gcc else os.path.splitext(os.path.basename(a.policy))[0]
    trace_path = f"{PROJECT}/docker_mnt/traffic_shell/trace_data/{a.trace}"
    n_trace = len(json.load(open(trace_path))["true_capacity"])
    n_steps = min(n_trace, a.max_steps) if a.max_steps else n_trace

    out_dir = os.path.join(PROJECT, "results_local", os.path.splitext(a.trace)[0],
                           f"{tag}_{datetime.now().strftime('%m%d%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    config = ENV_CONFIG
    # GCC: leave shm fields 0/1 at 0. Policy: drive bitrate + pacing rate.
    config["action_keys"] = ["prediction_bandwidth"] if is_gcc else ["bitrate", "pacing_rate"]
    config["action_limit"] = {}
    gs = config["gym_setting"]
    gs.update({"print_step": True, "print_period": 1.0, "duration": 1e6,
               "step_duration": .06, "observation_durations": [.06, .6],
               "history_size": 41, "logging_path": "/tmp/pandia.log",
               "skip_slow_start": 0, "enable_nvenc": False, "enable_nvdec": False,
               "action_cap": False})
    net_config = {"bw_file_name": a.trace, "delay": a.delay, "jitter": 0, "loss": 0}
    env = E.WebRTCEmulatorEnv_offline(config=config, net_config=net_config, curriculum_level=None)

    action = Action(config["action_keys"])
    obs_log, rx_log, tx_log, pred_log, cap_log, ts_log, rr_log = [], [], [], [], [], [], []
    est = 300e3
    try:
        env.reset()
        for i in range(n_steps):
            if not is_gcc:
                # Action.write() divides by K = 1024 to get "kbps"; compensate so
                # WebRTC receives estimate / 1000 kbps.
                action.bitrate = est * K / 1000
                action.pacing_rate = PACING_FACTOR * est * K / 1000
            env.step(action.array())
            obs_rx = rx.observation(time.monotonic())
            obs_tx = env.observation.array_bec().astype(np.float32)
            obs_raw = obs_rx if a.features == "receiver" else obs_tx
            ts_log.append(time.time() - env.start_ts)
            cap_log.append(env.obs_thread.cur_capacity * 1000.0)   # kbit -> bps
            obs_log.append(obs_raw.tolist())
            rx_log.append(obs_rx.tolist())
            tx_log.append(obs_tx.tolist())
            rr_log.append(float(obs_rx[5]))
            if is_gcc:
                pred_log.append(float("nan"))
            else:
                est = max(a.min_bps, policy(obs_raw))
                pred_log.append(est)
    except KeyboardInterrupt:
        pass
    env.close()
    print("receiver-side RTP records:", rx.n_records, flush=True)
    rx.close()

    # the estimate made at step i is compared with the capacity seen at step i
    rec = {"policy": a.policy, "trace": a.trace, "delay_ms": a.delay,
           "queue_ms": a.queue_ms, "audio": not a.no_audio, "features": a.features,
           "audio_interval_ms": a.audio_interval_ms,
           "pacing_factor": PACING_FACTOR, "observations": obs_log,
           "observations_rx": rx_log, "observations_tx": tx_log,
           "bandwidth_predictions": pred_log, "true_capacity": cap_log,
           "t": ts_log}
    with open(os.path.join(out_dir, a.trace), "w") as f:
        json.dump(rec, f)
    cap = np.asarray(cap_log)
    rr = np.asarray(rr_log)
    summary = {"policy": a.policy, "trace": a.trace, "steps": len(cap_log),
               "features": a.features, "queue_ms": a.queue_ms, "audio": not a.no_audio,
               "audio_interval_ms": a.audio_interval_ms,
               "utilisation": float(np.nansum(rr) / np.nansum(cap)) if np.nansum(cap) else None}
    if not is_gcc:
        summary.update(metrics(pred_log, cap_log))

    shutil.copy("/tmp/pandia.log", out_dir)
    generate_diagrams(out_dir, env.context, os.path.join(out_dir, "pandia.log"), cap / 1e6)
    for name in ("freeze.txt", "score.txt"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            summary[name] = open(p).read()
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith(".txt")}, indent=1))
    print("results in", out_dir)


if __name__ == "__main__":
    main()
