# Running RBWE's Pandia evaluation on a Windows PC without an NVIDIA GPU

Everything here is local glue; RBWE's code (`pandia/`, `containers/`, `docker_mnt/`) is unchanged.

## Images

| image | Dockerfile | content |
|---|---|---|
| `pandia-driver:local` | `Dockerfile.driver` | python 3.8 + torch 2.0.1 (CPU) + docker CLI; runs RBWE's Python side (AF_UNIX sockets do not work in Windows Python) |
| `pandia-emubase:local` | `Dockerfile.emubase` | driver + `iproute2` (tc) + `jq` |
| `pandia-emulator:local` | `Dockerfile.emulator` | emubase + `app/` + RBWE's `sb3_client.py` and `pandia/` + `nvstub/` |

- `app/` is the `/app` layer of `johnson163/pandia_emulator:latest` (blob `sha256:2e6768f8…`, 2024-02-03, verified): `simulation_Release` (WebRTC build after the 2024-01-09 pacer fix) and `media/drive_*p.yuv` (1440p/2160p and `simulation_Default` were left out). Only this 745 MB layer was downloaded; Docker Hub is ~0.2–0.5 MB/s here.
- Do **not** use `containers/emulator/simulation` from GitHub (2023-10-01): its pacer never sends media (`seen_first_packet_` bug).
- `nvstub/`: no-device stubs for `libcuda.so.1`, `libnvidia-encode.so.1`, `libnvcuvid.so.1` (the binary links them with BIND_NOW). With `NVENC`/`NVDEC` unset WebRTC uses OpenH264 / the software H.264 decoder.
- `simulation_video_save` (what RBWE's `sb3_client.py` starts) is a symlink to `simulation_Release`. RBWE's own build is not public; the public one ignores shm field 7 (`prediction_bandwidth`), so **the model output does not reach WebRTC and GCC controls the rate**.

## Run one trace

Traces go in `docker_mnt/traffic_shell/trace_data/` (MMSys'24 emulated-call JSON); `docker_mnt/media/drive_720p.yuv` must exist (copy from `app/media/`).

```bash
docker run --rm --name pandia-driver-run \
  -v /run/desktop/mnt/host/c/Users/kanif/work/Pandia:/data2/kj/Workspace/Pandia \
  -v /var/run/docker.sock:/var/run/docker.sock -v /tmp:/tmp \
  -e HOST_PANDIA=/run/desktop/mnt/host/c/Users/kanif/work/Pandia \
  -e PYTHONPATH=/data2/kj/Workspace/Pandia -e PYTHONUNBUFFERED=1 \
  -w /data2/kj/Workspace/Pandia pandia-driver:local \
  python localtest/run_rbwe_local.py 07488.json
```
(In Git Bash prefix with `MSYS_NO_PATHCONV=1`.) Results: `results_zte1/trace_<id>/…/` (plots, `score.txt`, `freeze.txt`, `<id>.json`).

## Known issues seen in the first run (07488, constant ~1 Mbps)

- `tc.sh` replays obs[35] (queuing delay) as netem delay (up to ~450 ms) and obs[105]×50 as loss (up to ~5 %) → GCC backs off from ~0.95 to ~0.03 Mbps although capacity is constant.
- "True capacity" is parsed from `tc` output without its unit (`1Mbit` → 1 kbit), so it is 0 for the first steps and ER/OER become inf.
- No frames are dumped to `res_video/` with the public binary, so `cal_vmaf.sh` has nothing to score.
- Video only, no audio track.
