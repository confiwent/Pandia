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

## Receiver-side features, audio flow, RTCP bypass (plan B)

- `emulator/rx_capture.py`: AF_PACKET on lo captures every RTP packet after the
  shaping qdisc (= arrival at the receiver) and forwards (arrival, abs-send-time,
  PT, seq, SSRC, size, padding-only) to the driver. All WebRTC video packets
  carry abs-send-time (extmap id 2); it is stamped on CLOCK_MONOTONIC, the same
  clock as the capture, so arrival - send is the one-way delay.
- `emulator/audio_flow.py`: Opus-like audio, one 100 B RTP packet (PT 111) per
  frame with abs-send-time, through the same bottleneck (`--no-audio` turns it
  off). The Pandia sender itself has only a video track. The frame interval is
  set per trace from its audio packet rate (median audio packets per 60 ms MI,
  snapped to 20/30/40/60 ms): 20 ms in 72, 60 ms in 43, 30 ms in 33 of the 148
  subset calls.
- `rx_features.py`: the 150-dim observation on the receiver side. Delay
  definitions recovered from identities that hold exactly in the emulated test
  set: with D = OWD - OWD(first packet) + 200 ms, delay = mean D - 200,
  min seen = min D over the call (data: 196-200), queuing = mean D - min seen,
  ratio = mean D / min D in the MI, avg-min diff = mean D - min D in the MI;
  loss ratio = lost / (lost + received) (exact). Interarrival is an approximate
  fit; jitter (std of gaps) cannot be identified from the data.
- `emulator/trace_replay.py`: root prio qdisc; RTCP (UDP payload byte 1 in
  200-207, offset 29) goes through a delay-only band, media through
  netem (delay + loss) -> tbf. Before, feedback queued behind the video on the
  shared lo bottleneck. Queue length `--queue-ms`, default 1000 ms: the
  per-call maximum queuing delay in the subset has median 240 ms, p90 955 ms,
  max 2754 ms (a lower bound of the buffer, the behaviour policies do not
  always fill it); with 300 / 600 ms the GCC runs on 07482 had their p99
  queuing delay cut at ~359 / 645 ms while the test set reaches 991 ms.

## WebRTC with an external-estimate input (shm field 7)

`webrtc/metaband-shm7.patch` on johnson-li/webrtc@pandia (bfc4443): in
`RtpTransportControllerSend::PostUpdates()`, shm field 7 > 0 (bps) replaces
GoogCC's target rate (kept: GoogCC's RTT/loss fields), so the BitrateAllocator,
the encoder and the pacer (2.5x, WebRTC's default factor) work as with GoogCC;
GoogCC probes are dropped and its congestion window is disabled
(`PANDIA_EXT_KEEP_CWND=1` / `--keep-cwnd` keeps it). Field 7 = 0 is plain GoogCC.
The NVENC `lib_dirs` became the gn arg `nv_lib_dir`.

Build (s126, 40 cores, ~3 min; source = gclient checkout in `/data4t/knw/pandia/webrtc_pandia`):
```bash
cd src && git am metaband-shm7.patch
buildtools/linux64/gn gen out/Release --args='is_debug=false rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true rtc_use_x11=false treat_warnings_as_errors=false nv_lib_dir="<dir with libcuda.so, libnvcuvid.so, libnvidia-encode.so>"'
ninja -C out/Release simulation          # ninja 1.11.1 from GitHub releases
```
`out/Release/simulation` -> `localtest/app/simulation_shm7` (md5 c3ad00d7…);
the emulator image links `/app/simulation_video_save` to it.

Check with `--policy const:600000` on 07488: "External bandwidth estimate
active: 600 kbps" in pandia.log, pacing rate 1500 kbps, encoder target
565–573 kbps (600 kbps minus transport overhead, as BitrateAllocator does
with GoogCC). `run_policy.py --control shm7` is the default; `--control
shm01` keeps the earlier encoder/pacer override.

## QoS metrics (`qos_metrics.py`, also run at the end of `run_policy.py` -> `qos.json`)

- throughput: receiving rate (video + audio) of the newest 60 ms MI, mean; utilisation = / mean capacity
- one-way packet delay: propagation delay + queuing delay of the MI (mean, p95)
- packet loss: sum lost / sum (lost + received) over the MIs; packet jitter: mean of the interarrival-std feature
- end-to-end frame delay: decoded utc - captured utc per frame (FrameCaptured,
  SendPacket seq -> frame id, Frame decoding acked in pandia.log), p50 / p95 / std
- freezes as WebRTC's getStats: decoded-frame interval > max(3 x avg, avg + 150 ms)
  (avg of the previous 30 intervals); freeze count and total freeze time / call time.
  Pandia's own "Freeze rate" (freeze.txt) is the share of frame intervals > 120 ms
  and drops intervals > 1 s, so long freezes are not counted.
- decoded fps, encoded video bitrate, median encoded resolution, mean QP.

## Closed-loop test subset

`docker_mnt/traffic_shell/subset148.txt`: 148 of the 9405 emulated test calls,
stratified by behaviour policy (v0-v4) x capacity constant/varying x loss
yes/no (20 strata, proportional), spread over mean capacity within each
stratum (seed 20261009). Copy the JSONs into `trace_data/` (not in git).
- `run_policy.py --features receiver` (default) feeds the receiver-side
  observation to the policy; both `observations_rx` and Pandia's sender-side
  `observations_tx` are saved.

GCC smoke test on 07482 (BBB + audio): audio / video share 0.43 / 0.57
(test set 0.33 / 0.67, sender side 0 / 1), 827 RTCP packets took the bypass.

## Known issues seen in the first run (07488, constant ~1 Mbps, RBWE's tc.sh)

- `tc.sh` replays obs[35] (queuing delay) as netem delay (up to ~450 ms) and obs[105]×50 as loss (up to ~5 %) → GCC backs off from ~0.95 to ~0.03 Mbps although capacity is constant.
- "True capacity" is parsed from `tc` output without its unit (`1Mbit` → 1 kbit), so it is 0 for the first steps and ER/OER become inf (fixed by the new replay).
- `pandia.constants` has K = 1024: `test_single` divides the capacity (kbit) by 1024 but the prediction (bps) by 1024², a 1000/1024 mismatch (~2.4 %) in its ER/MSE.
- No frames are dumped to `res_video/` with the public binary, so `cal_vmaf.sh` has nothing to score.
- Video only, no audio track.
