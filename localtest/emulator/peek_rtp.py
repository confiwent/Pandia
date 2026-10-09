"""Debug helper: print RTP header-extension ids / PTs seen on lo for a few seconds."""
import socket, struct, time, collections
s = socket.socket(socket.AF_PACKET, socket.SOCK_DGRAM, socket.htons(0x0800))
s.bind(("lo", 0))
seen = collections.Counter(); ext_ids = collections.Counter(); t0 = time.time()
while time.time() - t0 < 6:
    data, addr = s.recvfrom(65535)
    if addr[2] != socket.PACKET_HOST or data[9] != 17:
        continue
    ihl = (data[0] & 0xF) * 4
    p = data[ihl + 8:]
    if len(p) < 12 or (p[0] >> 6) != 2:
        continue
    if 192 <= p[1] <= 223:
        seen["rtcp"] += 1; continue
    pt = p[1] & 0x7F; cc = p[0] & 0xF; x = (p[0] >> 4) & 1
    seen[pt] += 1
    if x:
        off = 12 + 4 * cc
        prof, ln = struct.unpack("!HH", p[off:off + 4])
        e = p[off + 4: off + 4 + 4 * ln]; i = 0
        ids = []
        while i < len(e):
            b = e[i]
            if b == 0: i += 1; continue
            if prof == 0xBEDE:
                eid, l = b >> 4, (b & 0xF) + 1; ids.append(eid); i += 1 + l
            else:
                eid, l = e[i], e[i + 1]; ids.append(eid); i += 2 + l
        for eid in ids: ext_ids[(pt, hex(prof), eid)] += 1
print("PT counts", dict(seen)); print("ext ids", dict(ext_ids))
