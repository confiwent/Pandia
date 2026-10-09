"""
Replay an MMSys'24 emulated-call trace on the loopback interface with tc.

Replaces RBWE's tc.sh + get_tbf_rate():
  * capacity  = true_capacity (bps) / 1000 -> kbit, no clipping or skipped steps
  * loss      = true_loss_rate (already in %), applied by netem
  * delay     = a fixed one-way propagation delay (the dataset's delay features
                carry an unknown sender/receiver clock offset, so no per-step
                delay is replayed)
  * timing    = step i is applied at t0 + i * step (absolute schedule, no drift);
                tc is only called when a value changes
  * the true capacity of every step is reported to the observer socket
    (msg type 4, {"rate": kbit}) instead of parsing `tc -s qdisc show`

Qdisc layout on lo (sender, receiver and the audio flow all use lo):

    root prio (2 bands, everything -> band 2)
      band 1 (1:1): netem delay            <- RTCP (feedback), u32 filter on PT 200-207
      band 2 (1:2): netem delay + loss -> tbf rate   <- media (RTP), STUN/DTLS

so feedback is delayed like the media but neither shaped, queued behind the
video nor dropped (separate up/down links, as in a real path and in the
dataset's receiver-side estimator). tbf uses `latency` so the queue holds the
same time worth of data at any rate.
"""
import json
import math
import socket
import subprocess
import threading
import time

DEV = "lo"


def _tc(args):
    r = subprocess.run(["tc"] + args, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print("tc", " ".join(args), "->", r.stderr.decode().strip(), flush=True)


def _burst_bytes(kbit):
    # Small enough to keep shaping tight, large enough for one MTU and the
    # kernel timer granularity (rate / HZ) at high rates.
    return int(max(3000, kbit * 1000 / 8 * 0.005))


class TraceReplayer(threading.Thread):
    def __init__(self, trace_path, obs_socket_path, delay_ms=20.0,
                 step_s=0.06, queue_ms=300, log_path=None, min_kbit=10):
        super().__init__(daemon=True)
        with open(trace_path) as f:
            d = json.load(f)
        self.capacity_kbit = self._fill([c / 1000 if c is not None else math.nan
                                         for c in d["true_capacity"]], 1000.0)
        self.loss_pct = self._fill([x if x is not None else math.nan
                                    for x in d.get("true_loss_rate", [0] * len(d["true_capacity"]))], 0.0)
        self.delay_ms = delay_ms
        self.step_s = step_s
        self.queue_ms = queue_ms
        self.min_kbit = min_kbit
        self.log_path = log_path
        self.obs_socket_path = obs_socket_path
        self.stop_event = threading.Event()

    @staticmethod
    def _fill(values, first):
        out, last = [], first
        for v in values:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = last
            out.append(float(v))
            last = v
        return out

    def _netem_media(self, verb, loss):
        _tc(["qdisc", verb, "dev", DEV, "parent", "1:2", "handle", "20:", "netem",
             "delay", f"{self.delay_ms}ms", "loss", f"{loss}%"])

    def _tbf(self, verb, kbit):
        _tc(["qdisc", verb, "dev", DEV, "parent", "20:1", "handle", "30:", "tbf",
             "rate", f"{kbit}kbit", "burst", str(_burst_bytes(kbit)),
             "latency", f"{self.queue_ms}ms"])

    def _setup(self, kbit, loss):
        subprocess.run(["tc", "qdisc", "del", "dev", DEV, "root"], stderr=subprocess.DEVNULL)
        _tc(["qdisc", "add", "dev", DEV, "root", "handle", "1:", "prio", "bands", "2",
             "priomap"] + ["1"] * 16)
        _tc(["qdisc", "add", "dev", DEV, "parent", "1:1", "handle", "10:", "netem",
             "delay", f"{self.delay_ms}ms"])
        self._netem_media("add", loss)
        self._tbf("add", kbit)
        # RTCP: UDP payload byte 1 (packet type) in 200..207. The IP header on
        # lo has no options (20 B), so that byte is at offset 20 + 8 + 1 = 29.
        _tc(["filter", "add", "dev", DEV, "parent", "1:", "protocol", "ip", "prio", "1",
             "u32", "match", "ip", "protocol", "17", "0xff",
             "match", "u8", "0xc8", "0xf8", "at", "29", "flowid", "1:1"])

    def _report(self, sock, kbit):
        try:
            sock.send(bytes([4]) + json.dumps({"rate": kbit}).encode())
        except OSError:
            pass

    def run(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.connect(self.obs_socket_path)
        log = open(self.log_path, "w") if self.log_path else None
        if log:
            log.write("step,target_t,applied_t,capacity_kbit,loss_pct\n")
        n = len(self.capacity_kbit)
        last_kbit = last_loss = None
        t0 = time.monotonic()
        for i in range(n):
            if self.stop_event.is_set():
                break
            target = t0 + i * self.step_s
            wait = target - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            kbit = max(self.min_kbit, int(round(self.capacity_kbit[i])))
            loss = round(min(100.0, max(0.0, self.loss_pct[i])), 3)
            if i == 0:
                self._setup(kbit, loss)
            else:
                if loss != last_loss:
                    self._netem_media("change", loss)
                if kbit != last_kbit:
                    self._tbf("change", kbit)
            applied = time.monotonic()
            self._report(sock, kbit)
            if log:
                log.write(f"{i},{target - t0:.4f},{applied - t0:.4f},{kbit},{loss}\n")
            last_kbit, last_loss = kbit, loss
        if log:
            log.close()

    def stop(self):
        self.stop_event.set()
