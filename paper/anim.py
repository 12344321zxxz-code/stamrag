"""Shared animation state keyed off the timeline: lip-sync, blinks, Pip's ribbons, the tracker badge,
colour words, flying streamers and sparkles."""
import math
import os
from collections import defaultdict

import numpy as np
import skia

from . import cast as A
from . import script
from .craft import clamp, col, ease_back, ease_out, hash1, lerp, letter, path_of, piece, smoother, word_layout
from .core import BUILD, W
from .timeline import timeline

TL = timeline()
COLORS = script.COLORS
TRACKER_XY = (190, 168)
TRACKER_S = 0.95


def m(key):
    return TL.marks[key]


_MOUTH = None


def _mouths():
    global _MOUTH
    if _MOUTH is None:
        d = np.load(os.path.join(BUILD, "mouths.npz"))
        idx = defaultdict(list)
        for li, ln in enumerate(TL.lines):
            for mem in ln["members"]:
                idx[mem].append((ln["t0"] + ln["offsets"][mem], d[f"{li}:{mem}"]))
        _MOUTH = idx
    return _MOUTH


def mouth(member, t):
    for start, env in _mouths()[member]:
        k = (t - start) * 100
        if 0 <= k < len(env) - 1:
            i = int(k)
            return float(clamp((env[i] + (env[i + 1] - env[i]) * (k - i)) * 1.1))
    return 0.0


def speaking(member, t, pad=0.0):
    return any(s - pad <= t < s + len(e) / 100 + pad for s, e in _mouths()[member])


def blink(t, seed=0):
    period = 3.0 + 1.8 * hash1(seed + 3)
    ph = (t + hash1(seed + 7) * period) % period
    return max(0.0, 1 - abs(ph - 0.08) / 0.08) if ph < 0.16 else 0.0


# -- the colour hand-offs --------------------------------------------------------------------------------
def attach_time(color):
    if color == "green":
        return m("green.give") + 0.5
    if color == "purple":
        return m("purple.give") + 0.8
    return m(f"{color}.word") - 0.3


def ribbons(t):
    out = []
    launch = m("finale.launch")
    for i, c in enumerate(COLORS):
        amt = ease_out(clamp((t - attach_time(c)) / 0.35))
        if t >= launch + i * 0.08:
            amt = 0.0
        out.append((c, amt))
    return out


def tracker_state(t):
    fills, pops = [], []
    for c in COLORS:
        tw = m(f"{c}.word") + 1.0
        fills.append(ease_out(clamp((t - tw) / 0.18)))
        pops.append(t - tw)
    return fills, pops


def draw_tracker(c, t):
    t0, t1 = m("garden.tracker"), m("finale.launch")
    if t < t0:
        return
    a = clamp((t1 + 0.6 - t) / 0.6)
    s = TRACKER_S * ease_back(clamp((t - t0) / 0.45), 2.0)
    fills, pops = tracker_state(t)
    # arcs appear one by one when the badge arrives
    from .world import tracker
    tracker(c, TRACKER_XY[0], TRACKER_XY[1], max(0.01, s), fills, pops, t, alpha=a)


def star(c, x, y, r, alpha=1.0, rot=0.0, color="#fff6c8"):
    if alpha <= 0.02 or r <= 0.3:
        return
    pts = []
    for k in range(8):
        a = math.radians(rot + 45 * k)
        rr = r if k % 2 == 0 else r * 0.38
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    p = path_of(pts)
    c.drawPath(p, skia.Paint(AntiAlias=True, Color=col(color, alpha)))


def sparkle_trail(c, pts_fn, t, t0, t1, seed=0, n=10):
    """Sparkles left behind along a moving point: pts_fn(u) -> (x, y)."""
    for k in range(n):
        tk = t0 + (t1 - t0) * k / n
        age = t - tk
        if age < 0 or age > 0.6:
            continue
        x, y = pts_fn(clamp((tk - t0) / (t1 - t0)))
        x += 20 * (hash1(seed + k) - 0.5)
        y += 20 * (hash1(seed + k + 50) - 0.5) + age * 40
        star(c, x, y, 11 * (1 - age / 0.6), 1 - age / 0.6, rot=age * 200 + k * 30)


def bez(p0, p1, p2, u):
    return ((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0],
            (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1])


def flying_ribbon(c, color, x, y, t, rot=0.0, s=1.0, length=120, idx=0):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s, s)
    rp = A.ribbon_path(length / 2, 0, length, 12, t * 1.6, idx)
    piece(c, rp, color, 200 + idx, lift=2.0, rim=0.25, grad=0.1, shadow=0.3)
    c.restore()


def hand_off(c, color, t, p0, p2, t0, t1, lift_h=260, seed=0):
    """A streamer flying from p0 to p2 over [t0, t1] with a sparkle trail."""
    if t < t0 - 0.01 or t > t1 + 0.6:
        return
    p1 = ((p0[0] + p2[0]) / 2, min(p0[1], p2[1]) - lift_h)
    fn = lambda u: bez(p0, p1, p2, smoother(u))
    sparkle_trail(c, fn, t, t0, t1, seed)
    if t <= t1:
        u = clamp((t - t0) / (t1 - t0))
        x, y = fn(u)
        idx = COLORS.index(color)
        flying_ribbon(c, color, x, y, t, rot=math.sin(t * 7) * 15, s=0.9 + 0.3 * math.sin(math.pi * u), idx=idx)


# -- colour words -----------------------------------------------------------------------------------
def color_word(c, t):
    for i, colr in enumerate(COLORS):
        tw = m(f"{colr}.word")
        if not (tw - 0.05 <= t <= tw + 1.05):
            continue
        word = colr.upper()
        size = 170
        lay, width = word_layout(word, size)
        fly = smoother(clamp((t - tw - 0.72) / 0.3))
        cx = lerp(W / 2, TRACKER_XY[0], fly)
        cy = lerp(270, TRACKER_XY[1] + 10, fly)
        sc = lerp(1.0, 0.12, fly)
        for k, (ch, xo, adv) in enumerate(lay):
            u = clamp((t - tw - k * 0.05) / 0.28)
            if u <= 0:
                continue
            ls = ease_back(u, 2.4) * sc
            bob = 6 * math.sin(t * 6 + k) * (1 - fly)
            rot = (8 * (hash1(k + i * 10) - 0.5)) * (1 - fly) + 10 * math.sin(t * 3 + k) * 0.2
            x = cx + (xo - width / 2) * sc
            letter(c, ch, x, cy + bob, size, colr, seed=i * 20 + k, rot=rot, scale=max(0.01, ls),
                   lift=1.8 * sc + 0.2)


# -- Pip helpers ------------------------------------------------------------------------------------
def pip_draw(c, t, x, y, s=1.0, rot=0.0, mood="happy", look=None, hop=0.0, squash=0.0, bob=1.0, seed=0):
    by = 5 * math.sin(t * 1.9 + seed) * bob
    br = 2.2 * math.sin(t * 1.9 + 0.8 + seed) * bob
    mo = mouth("pip", t)
    if look is None:
        look = (0.35 * math.sin(t * 0.4), 0.0)
    A.pip(c, x, y + by - hop, s, rot + br, t, mouth_open=mo, look=look, blink=blink(t, 1), mood=mood,
          ribbons=ribbons(t), squash=squash)
    return x, y + by - hop, s, rot + br


def mast_point(pose, slot=0):
    x, y, s, r = pose
    return A.pip_point(x, y, s, r, -4, A.RIBBON_SLOTS[slot])
