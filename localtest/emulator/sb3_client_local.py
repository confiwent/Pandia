"""
Emulator-side client: RBWE's sb3_client with the trace replay replaced by
trace_replay.TraceReplayer (see that module for what changed and why).

The start command keeps RBWE's format {"bw": <trace file>, "delay": <ms>,
"loss": .., "jitter": ..}; "delay" is used as the fixed one-way propagation
delay, "loss" and "jitter" are ignored (loss comes from true_loss_rate).

Container environment:
  QUEUE_MS         bottleneck queue (tbf latency), default 1000
  AUDIO            1 (default) = run the Opus-like audio flow (audio_flow.py)
  AUDIO_INTERVAL_MS  audio frame interval, default 20
  PKT_SOCKET_PATH  if set, forward receiver-side RTP records there (rx_capture.py)
"""
import os
import socket
import subprocess
import time

import sb3_client as rbwe
from audio_flow import AudioFlow
from rx_capture import RxCapture
from trace_replay import TraceReplayer


class LocalClientProtocol(rbwe.ClientProtocol):
    def start_simulator(self, bw="02651.json", delay=20, loss=0, jitter=0):
        trace = "/app/traffic_shell/trace_data/" + bw
        log_path = f"{os.path.splitext(self.obs_socket_path)[0]}_replay.csv"
        print(f"Replay {trace}: delay {delay} ms, log {log_path}", flush=True)
        queue_ms = int(os.getenv("QUEUE_MS", "1000"))
        self.replayer = TraceReplayer(trace, self.obs_socket_path, delay_ms=float(delay),
                                      queue_ms=queue_ms, log_path=log_path)
        self.replayer.start()
        time.sleep(0.2)   # qdisc in place before any traffic
        self.extra = []
        if os.getenv("PKT_SOCKET_PATH"):
            self.extra.append(RxCapture(os.environ["PKT_SOCKET_PATH"]))
        if os.getenv("AUDIO", "1") == "1":
            self.extra.append(AudioFlow())
        for t in self.extra:
            t.start()
        print(f"audio interval {os.getenv('AUDIO_INTERVAL_MS', '20')} ms, queue {queue_ms} ms, extra threads: {[type(t).__name__ for t in self.extra]}", flush=True)

        self.sb3_logging_path = "/app/media/sb3.log"
        log_file = open(self.sb3_logging_path, "w")
        self.process = subprocess.Popen(
            ["/app/simulation_video_save",
             "--obs_socket", self.obs_socket_path,
             "--resolution", str(self.height), "--fps", str(self.fps),
             "--logging_path", self.logging_path,
             "--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/WebRTC-FrameDropper/Disabled",
             "--path", "/app/media",
             "--dump_path", "/app/media/res_video"],
            stdout=log_file, stderr=log_file, shell=False)

    def datagram_received(self, data: bytes, addr) -> None:
        if data[0] == 0:
            for t in [getattr(self, "replayer", None)] + getattr(self, "extra", []):
                if t is not None:
                    t.stop()
            self.replayer, self.extra = None, []
        super().datagram_received(data, addr)


def main():
    client = LocalClientProtocol()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    print(f"Connecting to {client.ctrl_socket_path}...", flush=True)
    sock.bind(client.ctrl_socket_path)
    os.chmod(client.ctrl_socket_path, 0o777)
    while True:
        data, addr = sock.recvfrom(1024)
        client.datagram_received(data, addr)


if __name__ == "__main__":
    main()
