"""
Emulator-side client: RBWE's sb3_client with the trace replay replaced by
trace_replay.TraceReplayer (see that module for what changed and why).

The start command keeps RBWE's format {"bw": <trace file>, "delay": <ms>,
"loss": .., "jitter": ..}; "delay" is used as the fixed one-way propagation
delay, "loss" and "jitter" are ignored (loss comes from true_loss_rate).

Container environment:
  QUEUE_MS         bottleneck queue (tbf latency), default 1000
  AUDIO            webrtc (default) = real Opus audio track inside WebRTC
                   (simulation --audio_wav, shares the bandwidth estimate with
                   the video through the BitrateAllocator); 1 = the external
                   Opus-like flow (audio_flow.py, outside the estimate); 0 = none
  AUDIO_INTERVAL_MS  audio frame length, default 20 (Opus: 20/40/60)
  AUDIO_WAV        audio source for AUDIO=webrtc, default /app/media/bbb_audio_48k.wav
  FRAME_DROPPER    1 = keep WebRTC's frame dropper (Pandia disables it)
  CAP_SCALE        shaping rate = CAP_SCALE x true_capacity (calibration), default 1
  EXTRA_FIELD_TRIALS  appended to --force_fieldtrials
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
        cap_scale = float(os.getenv("CAP_SCALE", "1"))
        self.replayer = TraceReplayer(trace, self.obs_socket_path, delay_ms=float(delay),
                                      queue_ms=queue_ms, log_path=log_path, cap_scale=cap_scale)
        print(f"capacity scale {cap_scale}", flush=True)
        self.replayer.start()
        time.sleep(0.2)   # qdisc in place before any traffic
        self.extra = []
        if os.getenv("PKT_SOCKET_PATH"):
            self.extra.append(RxCapture(os.environ["PKT_SOCKET_PATH"]))
        audio_mode = os.getenv("AUDIO", "webrtc")
        if audio_mode == "1":
            self.extra.append(AudioFlow())
        for t in self.extra:
            t.start()
        ptime = int(round(float(os.getenv("AUDIO_INTERVAL_MS", "20"))))
        print(f"audio {audio_mode}, frame {ptime} ms, queue {queue_ms} ms, extra threads: "
              f"{[type(t).__name__ for t in self.extra]}", flush=True)
        # Pandia disables WebRTC's frame dropper; FRAME_DROPPER=1 keeps it (WebRTC
        # default: lower frame rate at low bitrates, as real clients do).
        trials = "WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/"
        if os.getenv("FRAME_DROPPER", "0") != "1":
            trials += "WebRTC-FrameDropper/Disabled/"
        trials += os.getenv("EXTRA_FIELD_TRIALS", "")
        audio_args = []
        if audio_mode == "webrtc":
            audio_args = ["--audio_wav", os.getenv("AUDIO_WAV", "/app/media/bbb_audio_48k.wav"),
                          "--audio_ptime", str(ptime)]
            # Opus between 6 and 32 kbps (speech-like), allocated before the video
            trials += "WebRTC-Audio-Allocation/min:6kbps,max:32kbps/"

        self.sb3_logging_path = "/app/media/sb3.log"
        log_file = open(self.sb3_logging_path, "w")
        self.process = subprocess.Popen(
            ["/app/simulation_video_save",
             "--obs_socket", self.obs_socket_path,
             "--resolution", str(self.height), "--fps", str(self.fps),
             "--logging_path", self.logging_path,
             "--force_fieldtrials=" + trials,
             "--path", "/app/media",
             "--dump_path", "/app/media/res_video"] + audio_args,
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
