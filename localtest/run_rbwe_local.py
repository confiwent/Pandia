"""
Run RBWE's offline-model evaluation (pandia.agent.env_emulator_offline.test_single)
on a Windows + Docker Desktop host without an NVIDIA GPU, leaving the RBWE code
untouched. Adaptations made here:

  * the emulator container is started without `--runtime=nvidia --gpus all`,
    from the local image `pandia-emulator:local`, with the bind mounts mapped
    to this checkout (Docker Desktop host path given by HOST_PANDIA);
  * torch.load maps the RBWE critic checkpoint (saved on GPU) to the CPU.

Known limitation (see README_local.md): the public WebRTC build ignores the
`prediction_bandwidth` shared-memory field, so the model output does not reach
WebRTC and the call is effectively driven by GCC.

usage (inside pandia-driver:local, repo mounted at /data2/kj/Workspace/Pandia):
    python localtest/run_rbwe_local.py 07488.json
"""
import os
import sys

import torch

_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{"map_location": "cpu", **k})

import pandia.agent.env_emulator_offline as E  # noqa: E402

HOST_PANDIA = os.environ["HOST_PANDIA"]
IMAGE = os.environ.get("EMULATOR_IMAGE", "pandia-emulator:local")
# sb3_client_local = trace_replay.py; sb3_client = RBWE's original tc.sh replay
CLIENT = os.environ.get("EMULATOR_CLIENT", "sb3_client_local")


def start_container(self):
    cmd = (f'docker run -d --rm --name {self.container_name} '
           f'--hostname {self.container_name} '
           f'--cap-add=NET_ADMIN --cap-add=NET_RAW '
           f'-v /tmp:/tmp '
           f'-v {HOST_PANDIA}/docker_mnt/media:/app/media '
           f'-v {HOST_PANDIA}/docker_mnt/traffic_shell:/app/traffic_shell '
           f'--env PRINT_STEP=True -e SENDER_LOG=/tmp/sender.log --env BANDWIDTH=1000-3000 '
           f'--env OBS_SOCKET_PATH={self.obs_socket_path} '
           f'--env LOGGING_PATH={self.logging_path} '
           f'--env SB3_LOGGING_PATH={self.sb3_logging_path} '
           f'--env CTRL_SOCKET_PATH={self.ctrl_socket_path} '
           + ''.join(f'--env {k}={os.environ[k]} ' for k in ('QUEUE_MS', 'AUDIO', 'AUDIO_INTERVAL_MS', 'PKT_SOCKET_PATH')
                     if os.environ.get(k)) +
           f'{IMAGE} python -um {CLIENT}')
    print(cmd, flush=True)
    os.system(cmd)
    self.container = self.docker_client.containers.get(self.container_name)
    import time
    ts = time.time()
    while time.time() - ts < 10:
        try:
            self.control_socket.connect(self.ctrl_socket_path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.1)
    else:
        raise RuntimeError(f'Cannot connect to {self.ctrl_socket_path}')
    self.container_start_ts = time.time()


E.WebRTCEmulatorEnv_offline.start_container = start_container

if __name__ == "__main__":
    trace = sys.argv[1] if len(sys.argv) > 1 else "07488.json"
    E.test_single(trace, False, [1, 4])
