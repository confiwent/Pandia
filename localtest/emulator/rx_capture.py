"""
Receiver-side packet capture for the MMSys'24 features.

An AF_PACKET socket on lo sees every packet after it left the shaping qdisc,
i.e. at its arrival time at the receiver. For every RTP packet (RTCP, STUN and
DTLS are skipped) one record is forwarded to the driver over a UNIX datagram
socket:

    arrival   CLOCK_MONOTONIC seconds (same clock as WebRTC's rtc::TimeMillis,
              which stamps abs-send-time, so arrival - send = one-way delay)
    send_ast  24-bit abs-send-time (6.18 fixed-point seconds), 0xFFFFFFFF if absent
    pt, seq, ssrc
    size      RTP packet size in bytes (UDP payload)
    flags     bit 0: padding-only packet (probing)

Records are batched (<= 40 per datagram, flushed every 10 ms).
"""
import socket
import struct
import threading
import time

REC = struct.Struct("<dIBHIHB")
ABS_SEND_TIME_ID = 2          # a=extmap:2 .../abs-send-time in the Pandia SDP
NO_AST = 0xFFFFFFFF


def parse_rtp(p):
    """Return (pt, seq, ssrc, send_ast, padding_only) or None for non-RTP."""
    if len(p) < 12 or (p[0] >> 6) != 2 or 192 <= p[1] <= 223:
        return None
    cc, x, pad = p[0] & 0xF, (p[0] >> 4) & 1, (p[0] >> 5) & 1
    pt = p[1] & 0x7F
    seq, = struct.unpack_from("!H", p, 2)
    ssrc, = struct.unpack_from("!I", p, 8)
    off = 12 + 4 * cc
    ast = NO_AST
    if x and len(p) >= off + 4:
        prof, ln = struct.unpack_from("!HH", p, off)
        e0, e1 = off + 4, off + 4 + 4 * ln
        i = e0
        while i < e1:
            b = p[i]
            if b == 0:
                i += 1
                continue
            if prof == 0xBEDE:
                eid, l = b >> 4, (b & 0xF) + 1
                i += 1
            else:
                eid, l = b, p[i + 1]
                i += 2
            if eid == ABS_SEND_TIME_ID and l == 3:
                ast = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2]
            i += l
        off = e1
    padding_only = bool(pad) and len(p) > off and p[-1] >= len(p) - off
    return pt, seq, ssrc, ast, padding_only


class RxCapture(threading.Thread):
    def __init__(self, sink_path, dev="lo"):
        super().__init__(daemon=True)
        self.sink_path = sink_path
        self.dev = dev
        self.stop_event = threading.Event()

    def run(self):
        cap = socket.socket(socket.AF_PACKET, socket.SOCK_DGRAM, socket.htons(0x0800))
        cap.bind((self.dev, 0))
        cap.settimeout(0.01)
        out = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        out.connect(self.sink_path)
        buf, last_flush = [], time.monotonic()
        while not self.stop_event.is_set():
            try:
                data, addr = cap.recvfrom(65535)
                now = time.monotonic()
                if addr[2] == socket.PACKET_HOST and len(data) > 28 and data[9] == 17:
                    ihl = (data[0] & 0xF) * 4
                    p = data[ihl + 8:]
                    r = parse_rtp(p)
                    if r is not None:
                        pt, seq, ssrc, ast, padonly = r
                        buf.append(REC.pack(now, ast, pt, seq, ssrc, len(p), 1 if padonly else 0))
            except socket.timeout:
                now = time.monotonic()
            if buf and (len(buf) >= 40 or now - last_flush >= 0.01):
                try:
                    out.send(b"".join(buf))
                except OSError:
                    pass
                buf, last_flush = [], now

    def stop(self):
        self.stop_event.set()
