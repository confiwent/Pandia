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

## Trace replay (`emulator/trace_replay.py`, default)

The emulator runs `sb3_client_local.py`, which keeps RBWE's start command but
replaces `tc.sh` + `get_tbf_rate()`:

- capacity = `true_capacity` / 1000 kbit, no clipping or skipped steps (floor 10 kbit);
- loss = `true_loss_rate` (already in %, ramps in 0.6 % steps; matches the
  observed loss ratio, e.g. 4.78 % vs 0.0477 on 07488);
- fixed one-way delay = the start command's `delay` (20 ms in `test_single`);
  per-step delay is not replayed because the dataset's delay features carry an
  unknown clock offset (min-seen delay ~196 ms, some delays negative);
- step i is applied at t0 + i·60 ms (07482: apply lag p50 0.1 ms, p99 3.8 ms,
  no drift over 118 s); tc is called only on changes;
- tbf `latency 300ms` (queue holds 300 ms at any rate), burst max(3000 B, 5 ms);
- the true capacity is pushed to the observer socket every step;
- per-step log: `/tmp/<uuid>_obs_replay.csv` in the Docker VM.

`EMULATOR_CLIENT=sb3_client` switches back to RBWE's `tc.sh`.

Check on 07482 (0.22–1.6 Mbps, no loss), GCC in control: receiving rate never
above capacity, utilisation up to 0.94 in stable segments, queuing delay
peaks ~300 ms at capacity drops, overall utilisation 48.7 % (GCC ramps up
slowly after a drop), frame delay 52 ms, freeze rate (120 ms) 0.37 %.

## Closed-loop test of our own estimators (`run_policy.py`)

```bash
python localtest/run_policy.py 07482.json --policy localtest/models/miql_3c27_79200.onnx
python localtest/run_policy.py 07482.json --policy gcc
```
(same `docker run` wrapper as above). Models are not in git; copy them to
`localtest/models/` (MetaBand x3-maml-s1 `miql_3c27_79200.onnx`, the paper's
`sjtu-metaband.onnx` and `Schaferct_model.onnx` from s126).

- state: Pandia's raw `array_bec()` (MMSys'24 order, newest MI first, as in
  the dataset) × the offline evaluation's `NORMAL_VECTOR`, fed as `obs`
  [1,1,150]; recurrent inputs are zeros; estimate = `output[0,0,0]` bps;
- action: the public build's own shm fields, field 0 (encoder target) =
  estimate and field 1 (pacing rate) = 2.5 × estimate (WebRTC's default pacing
  factor); `Action.write()` divides by K = 1024, compensated in the script;
  `--policy gcc` leaves both at 0 (GCC in control);
- output `results_local/<trace>/<tag>_<time>/`: `<trace>.json` with per-step
  observations / bandwidth_predictions / true_capacity (MMSys'24 layout, bps),
  `summary.json` (ER / OER / MSE with the offline formulas, utilisation,
  freeze / score), Pandia's QoS plots.

Source video: Big Buck Bunny (Blender, `big_buck_bunny_720p_h264.mov`),
60–90 s, scaled to 1280×720 at 25 fps, raw I420 (`docker_mnt/media/drive_720p.yuv`,
750 frames, looped by WebRTC); made with `pandia-tools:local`
(`Dockerfile.tools`, ffmpeg + libvmaf; apt over HTTPS because the local proxy
returns 502 on plain-HTTP mirrors).

## Known issues seen in the first run (07488, constant ~1 Mbps, RBWE's tc.sh)

- `tc.sh` replays obs[35] (queuing delay) as netem delay (up to ~450 ms) and obs[105]×50 as loss (up to ~5 %) → GCC backs off from ~0.95 to ~0.03 Mbps although capacity is constant.
- "True capacity" is parsed from `tc` output without its unit (`1Mbit` → 1 kbit), so it is 0 for the first steps and ER/OER become inf (fixed by the new replay).
- `pandia.constants` has K = 1024: `test_single` divides the capacity (kbit) by 1024 but the prediction (bps) by 1024², a 1000/1024 mismatch (~2.4 %) in its ER/MSE.
- No frames are dumped to `res_video/` with the public binary, so `cal_vmaf.sh` has nothing to score.
- Video only, no audio track.
