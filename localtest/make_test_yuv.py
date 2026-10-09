"""Synthetic 1280x720 I420 clip for the Pandia sender (drive_720p.yuv).

Placeholder content for smoke tests only: a scrolling texture plus moving
blocks, 25 fps, looped by WebRTC's YUV frame generator.
"""
import sys
import numpy as np

W, H, FPS, SECONDS = 1280, 720, 25, 10
out = sys.argv[1]
rng = np.random.default_rng(0)
tex = (rng.random((H, 2 * W)) * 60).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:2 * W]
tex += 80 + 60 * np.sin(xx / 37.0) * np.cos(yy / 23.0)
with open(out, "wb") as f:
    for i in range(FPS * SECONDS):
        y = tex[:, (8 * i) % W:(8 * i) % W + W].copy()
        for k in range(4):
            cx = int((W - 160) * (0.5 + 0.5 * np.sin(i / 20.0 + k)))
            cy = int((H - 160) * (0.5 + 0.5 * np.cos(i / 27.0 + 2 * k)))
            y[cy:cy + 160, cx:cx + 160] = 40 + 50 * k
        u = np.full((H // 2, W // 2), 128 + 40 * np.sin(i / 30.0), np.float32)
        v = np.full((H // 2, W // 2), 128 + 40 * np.cos(i / 30.0), np.float32)
        for p in (y, u, v):
            f.write(np.clip(p, 0, 255).astype(np.uint8).tobytes())
print("frames", FPS * SECONDS)
