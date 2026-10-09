"""
A constant-rate, Opus-like audio flow sharing the shaped bottleneck with the
WebRTC video (the Pandia sender has no audio track).

One RTP packet (PT 111) every INTERVAL (20 ms by default, AUDIO_INTERVAL_MS
sets it per trace) with an abs-send-time extension stamped on
CLOCK_MONOTONIC, like WebRTC does: 12 B header + 8 B extension + 80 B payload
= 100 B (32 kbps of RTP). It is sent over UDP on lo to a sink that drains it,
so it is shaped, delayed and dropped like the video. It does not react to
congestion (as a fixed-rate audio stream).
"""
import os
import socket
import struct
import threading
import time

PT = 111
INTERVAL = 0.020
PAYLOAD = 80


class AudioFlow(threading.Thread):
    def __init__(self, port=47111, ssrc=0x0A0D10, interval=None):
        super().__init__(daemon=True)
        self.port, self.ssrc = port, ssrc
        self.interval = interval or float(os.getenv("AUDIO_INTERVAL_MS", "20")) / 1000
        self.stop_event = threading.Event()

    def run(self):
        sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sink.bind(("127.0.0.1", self.port))
        sink.setblocking(False)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = os.urandom(PAYLOAD)
        seq, ts = 0, 0
        t0 = time.monotonic()
        k = 0
        while not self.stop_event.is_set():
            target = t0 + k * self.interval
            wait = target - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            ast = int(time.monotonic() * (1 << 18)) & 0xFFFFFF
            hdr = struct.pack("!BBHII", 0x90, PT, seq & 0xFFFF, ts & 0xFFFFFFFF, self.ssrc)
            ext = struct.pack("!HH", 0xBEDE, 1) + bytes([(2 << 4) | 2,
                                                         (ast >> 16) & 0xFF, (ast >> 8) & 0xFF, ast & 0xFF])
            tx.sendto(hdr + ext + payload, ("127.0.0.1", self.port))
            seq += 1
            ts += int(48000 * self.interval)
            k += 1
            try:
                while True:
                    sink.recv(2048)
            except BlockingIOError:
                pass

    def stop(self):
        self.stop_event.set()
