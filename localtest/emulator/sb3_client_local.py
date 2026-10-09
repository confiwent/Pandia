"""
Emulator-side client: RBWE's sb3_client with the trace replay replaced by
trace_replay.TraceReplayer (see that module for what changed and why).

The start command keeps RBWE's format {"bw": <trace file>, "delay": <ms>,
"loss": .., "jitter": ..}; "delay" is used as the fixed one-way propagation
delay, "loss" and "jitter" are ignored (loss comes from true_loss_rate).
"""
import os
import socket
import subprocess
import time

import sb3_client as rbwe
from trace_replay import TraceReplayer


class LocalClientProtocol(rbwe.ClientProtocol):
    def start_simulator(self, bw="02651.json", delay=20, loss=0, jitter=0):
        trace = "/app/traffic_shell/trace_data/" + bw
        log_path = f"{os.path.splitext(self.obs_socket_path)[0]}_replay.csv"
        print(f"Replay {trace}: delay {delay} ms, log {log_path}", flush=True)
        self.replayer = TraceReplayer(trace, self.obs_socket_path,
                                      delay_ms=float(delay), log_path=log_path)
        self.replayer.start()

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
        if data[0] == 0 and getattr(self, "replayer", None) is not None:
            self.replayer.stop()
            self.replayer = None
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
