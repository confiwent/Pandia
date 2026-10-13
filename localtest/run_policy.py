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
How the estimate reaches WebRTC (--control):
  shm7  (default) field 7 = estimate in bps; our build (webrtc/metaband-shm7.patch)
        uses it in place of GoogCC's target rate, so BitrateAllocator, encoder
        and pacer (2.5x) work as with GoogCC
  shm01 Pandia's own fields: 0 (encoder target) = estimate, 1 (pacing rate) =
        PACING_FACTOR x estimate; this bypasses WebRTC's rate allocation
`--policy gcc` writes 0 everywhere, i.e. WebRTC's GoogCC is in control.

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
import sys
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
from qos_metrics import qos

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
    """input_norm 'nv': obs * NORMAL_VECTOR (our MetaBand models); 'raw': the raw
    observation, for models that normalise internally as the MMSys'24 challenge
    expects (Schaferct: ER 0.724 with NORMAL_VECTOR vs 0.386 raw on the subset)."""

    def __init__(self, path, input_norm="nv"):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.extras = {}
        for inp in self.sess.get_inputs():
            if inp.name != "obs":
                shape = [1 if (not isinstance(d, int) or d <= 0) else d for d in inp.shape]
                self.extras[inp.name] = np.zeros(shape, dtype=np.float32)
        self.nv = normal_vector() if input_norm == "nv" else np.ones(150, dtype=np.float32)

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


DRIFT_MEDIAN_PPM = 93.0     # median over the subset148 calls with an estimate


def trace_clock_drift_ppm(path):
    """Sender/receiver clock skew of a test-set call: Theil-Sen slope of the
    recorded short-MI queuing delay over time in MIs that do not congest (no
    loss, short-MI rate < 70 % of capacity), in ppm (1 ppm = 1e-3 ms/s),
    clipped to [0, 500]; the median of the subset when too few such MIs."""
    d = json.load(open(path))
    cap = np.asarray([np.nan if v is None else v for v in d["true_capacity"]], float)
    o = np.asarray(d["observations"], dtype=float)
    n = min(len(o), len(cap))
    o, cap = o[:n], cap[:n]
    ok = (o[:, 100] == 0) & (o[:, 0] < 0.7 * cap) & (o[:, 10] >= 2)
    ok[:100] = False
    idx = np.flatnonzero(ok)
    if len(idx) < 150:
        return DRIFT_MEDIAN_PPM
    t, q = idx * 0.06, o[idx, 30]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(t), 4000), rng.integers(0, len(t), 4000)
    m = np.abs(t[i] - t[j]) > 10
    slope = np.median((q[i][m] - q[j][m]) / (t[i][m] - t[j][m]))   # ms per s
    return float(np.clip(slope * 1000, 0, 500))


def trace_loss_burst(path):
    """Mean loss burst (packets) of a test-set call: the recorded "lost packets
    per loss event" (obs[110]) of its lossy 60 ms MIs over what independent
    loss at the call's mean true_loss_rate would give for the same packet
    counts; below 1.3 the call is treated as independent loss (1)."""
    d = json.load(open(path))
    tl = np.asarray([np.nan if v is None else v for v in d.get("true_loss_rate", [0])], float)
    q = np.nanmean(tl) / 100 if np.any(tl > 0) else 0.0
    o = np.asarray(d["observations"], dtype=float)[80:]
    m = o[:, 100] > 0
    if q <= 0 or m.sum() < 20:
        return 1.0
    p = o[m, 100]
    n = np.clip(np.round(o[m, 10] / np.maximum(1 - p, 1e-9)), 1, None)   # received + lost
    ratio = o[m, 110].mean() / np.mean(n * q / (1 - (1 - q) ** n))
    return float(np.clip(ratio, 1, 10)) if ratio >= 1.3 else 1.0


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
    ap.add_argument("--policy", required=True,
                    help="path to .onnx, 'gcc', 'const:<bps>' (fixed estimate, for tests) or "
                         "'replay' (the trace's own bandwidth_predictions, i.e. the "
                         "test-set behaviour policy, step by step: environment calibration)")
    ap.add_argument("--delay", type=float, default=20, help="one-way delay, ms")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = whole trace")
    ap.add_argument("--tag", default="", help="results_local/<tag>/<trace>/... (batch name)")
    ap.add_argument("--plots", action="store_true",
                    help="also write Pandia's ~30 PNG diagrams (off: they took 14 GB for 1.8k runs)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="do not gzip pandia.log and <trace>.json after the QoS is computed")
    ap.add_argument("--min-bps", type=float, default=20e3)
    ap.add_argument("--features", choices=["receiver", "sender"], default="receiver")
    ap.add_argument("--control", choices=["shm7", "shm01"], default="shm7")
    ap.add_argument("--alloc-audio", action="store_true",
                    help="deployment fix F1: give WebRTC estimate - measured audio rate "
                         "(the audio flow is outside WebRTC's BitrateAllocator)")
    ap.add_argument("--smooth", default="",
                    help="deployment fix F3 'a_up,a_down,max_up': asymmetric smoothing of the "
                         "estimate against the value applied at the previous step (follow "
                         "decreases with gain a_down, increases with gain a_up, and rise by at "
                         "most max_up per 60 ms step)")
    ap.add_argument("--guard", default="",
                    help="deployment fix F2 'q_ms,factor[,long]': if the queuing delay of the "
                         "newest short MI (60 ms; with ',long' the newest long MI, 600 ms, i.e. "
                         "a sustained queue) > q_ms, cap the estimate at factor x long-MI "
                         "receiving rate")
    ap.add_argument("--input-norm", choices=["auto", "nv", "raw"], default="auto",
                    help="auto: raw for Schaferct*, NORMAL_VECTOR otherwise")
    ap.add_argument("--keep-cwnd", action="store_true",
                    help="shm7: keep GoogCC's congestion window (PANDIA_EXT_KEEP_CWND)")
    ap.add_argument("--queue-ms", type=int, default=1000, help="bottleneck queue (tbf latency)")
    ap.add_argument("--audio", choices=["webrtc", "flow", "none"], default="webrtc",
                    help="webrtc: real Opus track inside WebRTC (shares the estimate); "
                         "flow: external Opus-like flow (outside the estimate); none")
    ap.add_argument("--no-audio", action="store_true", help="same as --audio none")
    ap.add_argument("--frame-dropper", action="store_true",
                    help="keep WebRTC's frame dropper (Pandia disables it)")
    ap.add_argument("--field-trials", default="", help="extra WebRTC field trials")
    ap.add_argument("--cap-scale", type=float, default=1.0,
                    help="shaping rate = cap_scale x true_capacity (environment calibration)")
    ap.add_argument("--audio-interval-ms", type=float, default=0,
                    help="audio frame interval; 0 = from the trace's audio packet rate")
    ap.add_argument("--tbf-burst", default="3000,5",
                    help="tbf bucket 'min_bytes,ms': max(min_bytes, ms of the rate)")
    ap.add_argument("--clock-drift", default="0",
                    help="receiver clock skew in ppm added to the delay features; "
                         "auto = estimated from the trace's recorded queuing delay")
    ap.add_argument("--loss-burst", default="1",
                    help="mean loss burst in packets (Gilbert model); 1 = independent loss; "
                         "auto = from the trace's recorded loss pattern")
    a = ap.parse_args()
    if not a.audio_interval_ms:
        a.audio_interval_ms = trace_audio_interval_ms(
            f"{PROJECT}/docker_mnt/traffic_shell/trace_data/{a.trace}")
    a.loss_burst = (trace_loss_burst(f"{PROJECT}/docker_mnt/traffic_shell/trace_data/{a.trace}")
                    if a.loss_burst == "auto" else float(a.loss_burst))
    a.clock_drift = (trace_clock_drift_ppm(f"{PROJECT}/docker_mnt/traffic_shell/trace_data/{a.trace}")
                     if a.clock_drift == "auto" else float(a.clock_drift))
    print(f"loss burst {a.loss_burst:.2f} packets, clock drift {a.clock_drift:.0f} ppm", flush=True)

    if a.keep_cwnd:
        os.environ["PANDIA_EXT_KEEP_CWND"] = "1"
    if a.frame_dropper:
        os.environ["FRAME_DROPPER"] = "1"
    os.environ["CAP_SCALE"] = str(a.cap_scale)
    burst_min, burst_ms = a.tbf_burst.split(",")
    os.environ.update({"TBF_BURST_MIN": str(int(burst_min)), "TBF_BURST_MS": str(float(burst_ms)),
                       "LOSS_BURST": str(a.loss_burst)})
    if a.field_trials:
        os.environ["EXTRA_FIELD_TRIALS"] = a.field_trials
    job_id = uuid.uuid4().hex[:8]
    pkt_sock = f"/tmp/{job_id}_pkt.sock"
    webrtc_log = f"/tmp/{job_id}_pandia.log"      # per job: batches run in parallel
    if a.no_audio:
        a.audio = "none"
    os.environ.update({"QUEUE_MS": str(a.queue_ms),
                       "AUDIO": {"webrtc": "webrtc", "flow": "1", "none": "0"}[a.audio],
                       "AUDIO_INTERVAL_MS": str(a.audio_interval_ms),
                       "PKT_SOCKET_PATH": pkt_sock})
    rx = RxFeatures(pkt_sock, drift_ppm=a.clock_drift)

    is_gcc = a.policy == "gcc"
    if is_gcc:
        policy, tag = None, "gcc"
    elif a.policy == "replay":
        behaviour = np.asarray(json.load(open(
            f"{PROJECT}/docker_mnt/traffic_shell/trace_data/{a.trace}"))["bandwidth_predictions"], float)
        replay_step = [0]

        def policy(obs):           # estimate for the next step
            replay_step[0] += 1
            return float(behaviour[min(replay_step[0], len(behaviour) - 1)])
        tag = "replay"
    elif a.policy.startswith("const:"):
        const_bps = float(a.policy.split(":", 1)[1])
        policy, tag = (lambda obs: const_bps), f"const{int(const_bps)}"
    else:
        if a.input_norm == "auto":
            a.input_norm = "raw" if "schaferct" in os.path.basename(a.policy).lower() else "nv"
        policy, tag = OnnxPolicy(a.policy, a.input_norm), os.path.splitext(os.path.basename(a.policy))[0]
    trace_path = f"{PROJECT}/docker_mnt/traffic_shell/trace_data/{a.trace}"
    n_trace = len(json.load(open(trace_path))["true_capacity"])
    n_steps = min(n_trace, a.max_steps) if a.max_steps else n_trace

    out_dir = os.path.join(PROJECT, "results_local", a.tag, os.path.splitext(a.trace)[0],
                           f"{tag}_{datetime.now().strftime('%m%d%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    config = ENV_CONFIG
    # GCC: every shm field 0. Policy: field 7 (shm7) or fields 0/1 (shm01).
    config["action_keys"] = (["prediction_bandwidth"] if is_gcc or a.control == "shm7"
                             else ["bitrate", "pacing_rate"])
    config["action_limit"] = {}
    gs = config["gym_setting"]
    gs.update({"print_step": True, "print_period": 1.0, "duration": 1e6,
               "step_duration": .06, "observation_durations": [.06, .6],
               "history_size": 41, "logging_path": webrtc_log,
               "skip_slow_start": 0, "enable_nvenc": False, "enable_nvdec": False,
               "action_cap": False})
    net_config = {"bw_file_name": a.trace, "delay": a.delay, "jitter": 0, "loss": 0}
    env = E.WebRTCEmulatorEnv_offline(config=config, net_config=net_config, curriculum_level=None)

    action = Action(config["action_keys"])
    obs_log, rx_log, tx_log, pred_log, cap_log, ts_log, rr_log = [], [], [], [], [], [], []
    qtrue_log = []       # newest short-MI queuing delay without the clock skew (for the QoS)
    applied_log = []     # what was written to WebRTC after F1/F2 (pred_log = raw model output)
    est = (float(a.policy.split(":", 1)[1]) if a.policy.startswith("const:")
           else float(behaviour[0]) if a.policy == "replay" else 300e3)
    try:
        env.reset()
        for i in range(n_steps):
            if is_gcc:
                action.prediction_bandwidth = 0
            elif a.control == "shm7":
                action.prediction_bandwidth = est          # bps, written as is
            else:
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
            qtrue_log.append(rx.true_queue)
            tx_log.append(obs_tx.tolist())
            rr_log.append(float(obs_rx[5]))
            if is_gcc:
                pred_log.append(float("nan"))
            else:
                raw = max(a.min_bps, policy(obs_raw))
                pred_log.append(raw)
                est = raw
                if a.smooth and applied_log:
                    a_up, a_down, max_up = (float(v) for v in a.smooth.split(","))
                    prev = applied_log[-1]
                    est = prev + (a_down if raw < prev else a_up) * (raw - prev)
                    est = min(est, prev * (1 + max_up))
                if a.guard:
                    g = a.guard.split(",")
                    q_ms, factor = float(g[0]), float(g[1])
                    q_now = obs_raw[35] if len(g) > 2 and g[2] == "long" else obs_raw[30]
                    if q_now > q_ms:
                        est = max(a.min_bps, min(est, factor * float(obs_raw[5])))
                if a.alloc_audio:
                    audio_bps = float(obs_raw[5] * obs_raw[135])     # long MI: rate x audio share
                    est = max(a.min_bps, est - audio_bps)
                applied_log.append(est)
    except KeyboardInterrupt:
        pass
    env.close()
    print("receiver-side RTP records:", rx.n_records, flush=True)
    rx.close()

    # the estimate made at step i is compared with the capacity seen at step i
    rec = {"policy": a.policy, "trace": a.trace, "delay_ms": a.delay,
           "queue_ms": a.queue_ms, "audio": a.audio, "features": a.features,
           "control": a.control, "keep_cwnd": a.keep_cwnd, "input_norm": a.input_norm,
           "frame_dropper": a.frame_dropper, "field_trials": a.field_trials, "cap_scale": a.cap_scale,
           "audio_interval_ms": a.audio_interval_ms, "tbf_burst": a.tbf_burst, "loss_burst": a.loss_burst,
           "clock_drift_ppm": a.clock_drift, "queue_true_ms": qtrue_log,
           "pacing_factor": PACING_FACTOR, "observations": obs_log,
           "observations_rx": rx_log, "observations_tx": tx_log,
           "applied_estimate": applied_log, "alloc_audio": a.alloc_audio, "guard": a.guard,
           "smooth": a.smooth,
           "bandwidth_predictions": pred_log, "true_capacity": cap_log,
           "t": ts_log, "tag": a.tag}
    with open(os.path.join(out_dir, a.trace), "w") as f:
        json.dump(rec, f)
    cap = np.asarray(cap_log)
    rr = np.asarray(rr_log)
    summary = {"policy": a.policy, "trace": a.trace, "steps": len(cap_log),
               "features": a.features, "queue_ms": a.queue_ms, "audio": a.audio,
               "control": a.control, "keep_cwnd": a.keep_cwnd,
               "audio_interval_ms": a.audio_interval_ms,
               "utilisation": float(np.nansum(rr) / np.nansum(cap)) if np.nansum(cap) else None}
    if not is_gcc:
        summary.update(metrics(pred_log, cap_log))

    shutil.copy(webrtc_log, os.path.join(out_dir, "pandia.log"))
    os.remove(webrtc_log)
    if a.plots:
        try:   # Pandia's plots; they fail on some runs (negative frame times, no decoded frame)
            generate_diagrams(out_dir, env.context, os.path.join(out_dir, "pandia.log"), cap / 1e6)
        except Exception as e:
            summary["diagrams_error"] = repr(e)
    for name in ("freeze.txt", "score.txt"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            summary[name] = open(p).read()
    try:
        summary["qos"] = qos(out_dir)        # deployment metrics, also in qos.json
    except Exception as e:                   # keep the run even if parsing fails
        summary["qos_error"] = repr(e)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if not a.keep_raw and "qos" in summary:
        from compress_results import compress
        for name in ("pandia.log", a.trace):
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                compress(path)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith(".txt")}, indent=1))
    print("results in", out_dir)


if __name__ == "__main__":
    main()
    # Leftover threads (observer socket, docker client) can keep the process
    # alive after the results are written; a batch then waits forever.
    sys.stdout.flush()
    os._exit(0)
