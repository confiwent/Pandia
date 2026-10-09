"""
Receiver-side MMSys'24 observation (15 features x 5 short 60 ms MIs + 5 long
600 ms MIs = 150), computed from the RTP records sent by emulator/rx_capture.py.

Layout as in the dataset: feature f occupies [10f, 10f+10), short MIs first,
newest MI first. At step time T the short MI k covers [T-(k+1)·60 ms, T-k·60 ms)
and the long MI k covers [T-(k+1)·600 ms, T-k·600 ms).

Per-packet quantities
  delay      one-way delay (ms): arrival (CLOCK_MONOTONIC) - abs-send-time,
             both on WebRTC's clock, modulo the 64 s abs-send-time wrap
  lost       sequence-number gaps per SSRC, counted at the packet that reveals them
  kind       audio = PT 111 (audio_flow.py), probe = padding-only, video = rest

The MMSys'24 extractor is not public; the delay definitions below were
recovered from the identities that hold exactly in the emulated test set
(e.g. queuing = delay - min_seen + 200, ratio = (delay+200)/(delay-amd+200)).
With D = one-way delay - one-way delay of the first packet + 200 ms:
   0 receiving rate (bps)        bytes * 8 / MI length          (exact in data)
   1 number of received packets
   2 received bytes
   3 queuing delay (ms)          mean(D) - min seen D            (exact)
   4 delay (ms)                  mean(D) - 200                   (exact)
   5 minimum seen delay (ms)     min of D over the call so far   (data: 196-200)
   6 delay ratio                 mean(D) / min(D in the MI)      (exact)
   7 delay avg-min difference    mean(D) - min(D in the MI)      (exact)
   8 packet interarrival (ms)    mean gap of each packet in the MI to the
                                 previous packet (approximate fit)
   9 packet jitter (ms)          std of those gaps (not identifiable from the data)
  10 packet loss ratio           lost / (lost + received)        (exact)
  11 number of lost packets      lost
  12-14 video / audio / probing packet share
The test set has no MI without packets (audio every 20 ms); here such an MI
gets zeros, except the delay features (3-7), which keep the last value.
"""
import bisect
import os
import socket
import struct
import threading
from collections import defaultdict

import numpy as np

REC = struct.Struct("<dIBHIHB")
NO_AST = 0xFFFFFFFF
AUDIO_PT = 111
AST_WRAP = 1 << 24
SHORT, LONG = 0.06, 0.6


class RxFeatures:
    def __init__(self, sock_path):
        self.sock_path = sock_path
        if os.path.exists(sock_path):
            os.remove(sock_path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(sock_path)
        os.chmod(sock_path, 0o777)
        self.lock = threading.Lock()
        # per packet: arrival, size, delay_ms (nan if unknown), kind (0 video 1 audio 2 probe)
        self.t, self.size, self.delay, self.kind = [], [], [], []
        self.loss_t, self.loss_n = [], []
        self.highest = {}
        self.min_seen = np.inf      # min of D
        self.base = None            # one-way delay of the first packet
        self.last_delay = None
        self.n_records = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._rx, daemon=True)
        self.thread.start()

    def _rx(self):
        self.sock.settimeout(0.2)
        while not self.stop_event.is_set():
            try:
                data = self.sock.recv(65535)
            except socket.timeout:
                continue
            except OSError:          # socket closed by close()
                break
            with self.lock:
                for arrival, ast, pt, seq, ssrc, size, flags in REC.iter_unpack(data):
                    self._add(arrival, ast, pt, seq, ssrc, size, flags)

    def _add(self, arrival, ast, pt, seq, ssrc, size, flags):
        self.n_records += 1
        if ast != NO_AST:
            now_ast = int(arrival * (1 << 18)) % AST_WRAP
            d = ((now_ast - ast) % AST_WRAP) / (1 << 18) * 1000.0
            if d > 30000:           # sent "after" arrival by < 1 tick: treat as ~0
                d -= AST_WRAP / (1 << 18) * 1000.0
            if self.base is None:
                self.base = d
            d = d - self.base + 200.0   # D
            self.min_seen = min(self.min_seen, d)
        else:
            d = np.nan
        kind = 2 if flags & 1 else (1 if pt == AUDIO_PT else 0)
        self.t.append(arrival)
        self.size.append(size)
        self.delay.append(d)
        self.kind.append(kind)
        # loss from sequence gaps (extended 16-bit sequence numbers)
        h = self.highest.get(ssrc)
        if h is None:
            self.highest[ssrc] = seq
        else:
            ext = h + ((seq - (h & 0xFFFF) + 0x8000) % 0x10000 - 0x8000)
            if ext > h:
                if ext > h + 1:
                    self.loss_t.append(arrival)
                    self.loss_n.append(ext - h - 1)
                self.highest[ssrc] = ext

    def _window(self, a, b, carry):
        i0, i1 = bisect.bisect_left(self.t, a), bisect.bisect_left(self.t, b)
        l0, l1 = bisect.bisect_left(self.loss_t, a), bisect.bisect_left(self.loss_t, b)
        lost = float(sum(self.loss_n[l0:l1]))
        n = i1 - i0
        f = np.zeros(15, dtype=np.float64)
        if n == 0:
            f[3:8] = carry
            f[10] = 1.0 if lost > 0 else 0.0
            f[11] = lost
            return f
        size = np.asarray(self.size[i0:i1], float)
        kind = np.asarray(self.kind[i0:i1])
        d = np.asarray(self.delay[i0:i1], float)
        d = d[~np.isnan(d)]
        t = np.asarray(self.t[i0:i1])
        f[0] = size.sum() * 8 / (b - a)
        f[1] = n
        f[2] = size.sum()
        if len(d) and np.isfinite(self.min_seen):
            m, mn = d.mean(), d.min()
            f[3], f[4], f[5] = m - self.min_seen, m - 200.0, self.min_seen
            f[6], f[7] = m / max(mn, 1e-3), m - mn
        else:
            f[3:8] = carry
        prev = self.t[i0 - 1:i1] if i0 > 0 else self.t[i0:i1]
        if len(prev) > 1:
            gaps = np.diff(np.asarray(prev)) * 1000
            f[8], f[9] = gaps.mean(), gaps.std()
        f[10] = lost / (lost + n)
        f[11] = lost
        f[12] = np.mean(kind == 0)
        f[13] = np.mean(kind == 1)
        f[14] = np.mean(kind == 2)
        return f

    def observation(self, now):
        """150-dim observation at monotonic time `now`."""
        with self.lock:
            carry = np.zeros(5) if self.last_delay is None else self.last_delay
            short = [self._window(now - (k + 1) * SHORT, now - k * SHORT, carry) for k in range(5)]
            long_ = [self._window(now - (k + 1) * LONG, now - k * LONG, carry) for k in range(5)]
            # carry forward the newest delay features that came from real packets
            for w in short:
                if w[1] > 0 and w[5] != 0:
                    self.last_delay = w[3:8].copy()
                    break
        obs = np.zeros(150, dtype=np.float32)
        for fi in range(15):
            obs[10 * fi:10 * fi + 5] = [w[fi] for w in short]
            obs[10 * fi + 5:10 * fi + 10] = [w[fi] for w in long_]
        return obs

    def close(self):
        self.stop_event.set()
        self.sock.close()
        if os.path.exists(self.sock_path):
            os.remove(self.sock_path)
