"""Debug helper: per-PT packet rate / size / ext ids on lo for N seconds (after shaping)."""
import socket, struct, sys, time, collections
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 8
s = socket.socket(socket.AF_PACKET, socket.SOCK_DGRAM, socket.htons(0x0800)); s.bind(("lo", 0))
cnt = collections.Counter(); size = collections.defaultdict(list); exts = collections.defaultdict(set); t0 = time.time()
while time.time() - t0 < dur:
    d, a = s.recvfrom(65535)
    if a[2] != socket.PACKET_HOST or d[9] != 17: continue
    p = d[(d[0] & 15) * 4 + 8:]
    if len(p) < 12 or p[0] >> 6 != 2 or 192 <= p[1] <= 223: continue
    pt = p[1] & 127; cnt[pt] += 1; size[pt].append(len(p))
    if (p[0] >> 4) & 1:
        off = 12 + 4 * (p[0] & 15); ln = struct.unpack("!H", p[off + 2:off + 4])[0]; e = p[off + 4:off + 4 + 4 * ln]; i = 0
        while i < len(e):
            if e[i] == 0: i += 1; continue
            exts[pt].add(e[i] >> 4); i += 1 + (e[i] & 15) + 1
for pt in sorted(cnt):
    z = sorted(size[pt]); print(f"PT {pt:3d}: {cnt[pt]/dur:6.1f} pkt/s, size median {z[len(z)//2]} B, {sum(z)*8/dur/1e3:7.1f} kbps, ext ids {sorted(exts[pt])}")
