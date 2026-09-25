"""Timing + shared helpers for the CHASE clip (drawing helpers are re-used from mv.core)."""
import math
import os

import numpy as np
import skia

from mv.core import (BM, FONT_DIR, H, TAU, W, C, P, clamp, col, ease_back, ease_elastic, ease_in, ease_io, ease_out,
                     fbm, font, fract, hash1, hit, hsv, hx, inv_lerp, lerp, lin_grad, mix, poly, pulse, rad_grad,
                     remap, rot, scale_c, smooth, smoother, tri, vnoise)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build_chase")
FPS = 30

__all__ = ["BM", "FONT_DIR", "H", "TAU", "W", "C", "P", "clamp", "col", "ease_back", "ease_elastic", "ease_in",
           "ease_io", "ease_out", "fbm", "font", "fract", "hash1", "hit", "hsv", "hx", "inv_lerp", "lerp", "lin_grad",
           "mix", "poly", "pulse", "rad_grad", "remap", "rot", "scale_c", "smooth", "smoother", "tri", "vnoise",
           "ROOT", "BUILD", "FPS", "audio", "Audio", "circle_path", "rrect", "bez_q", "path_from"]


class Audio:
    def __init__(self, path):
        d = np.load(path)
        self.grid = d["grid"]
        self.n_before = int(d["n_before"])
        self.duration = float(d["duration"])
        self.ft = d["ft"]
        self.env = {k: d[k] for k in ("kick", "snare", "hat", "bass", "gtr", "rms", "onset", "speech")}
        self.spec = d["spec"]
        self.dive_t, self.dive_m = d["dive_t"], d["dive_m"]
        self.notes = d["notes"]          # (start, end, midi, amp)
        self.lead = d["lead"]
        self.phrases = d["phrases"]
        self.period = float(np.median(np.diff(self.grid)))
        self.note_starts = self.notes[:, 0]
        self.lead_starts = self.lead[:, 0]

    # beats are numbered so that beat 0 = the band's first downbeat; negative before it
    def beat(self, t):
        g = self.grid
        k = np.arange(len(g)) - self.n_before
        if t <= g[0]:
            return k[0] + (t - g[0]) / self.period
        if t >= g[-1]:
            return k[-1] + (t - g[-1]) / self.period
        return float(np.interp(t, g, k))

    def beat_time(self, b):
        g = self.grid
        k = np.arange(len(g)) - self.n_before
        if b <= k[0]:
            return g[0] + (b - k[0]) * self.period
        if b >= k[-1]:
            return g[-1] + (b - k[-1]) * self.period
        return float(np.interp(b, k, g))

    def bar_time(self, bar):
        return self.beat_time(bar * 4.0)

    def e(self, name, t):
        return float(np.interp(t, self.ft, self.env[name]))

    def bands(self, t):
        i = int(np.clip(np.searchsorted(self.ft, t), 0, len(self.ft) - 1))
        return self.spec[i]

    def dive(self, t):
        return float(np.interp(t, self.dive_t, self.dive_m))

    def notes_in(self, t0, t1, lead=False):
        arr = self.lead if lead else self.notes
        st = self.lead_starts if lead else self.note_starts
        i0, i1 = np.searchsorted(st, t0), np.searchsorted(st, t1)
        return arr[i0:i1]

    def last_note(self, t, lead=True, min_pitch=0):
        arr = self.lead if lead else self.notes
        st = self.lead_starts if lead else self.note_starts
        i = np.searchsorted(st, t) - 1
        while i >= 0 and arr[i, 2] < min_pitch:
            i -= 1
        return arr[i] if i >= 0 else None

    def pitch_at(self, t, default=55.0):
        """Smoothed lead pitch (MIDI) at time t: glides between successive lead notes."""
        st = self.lead_starts
        i = np.searchsorted(st, t) - 1
        if i < 0:
            return default
        p0 = self.lead[i, 2]
        if i + 1 < len(st):
            t1 = st[i + 1]
            glide = clamp((t - (t1 - 0.06)) / 0.06)
            return lerp(p0, self.lead[i + 1, 2], glide)
        return p0


_audio = None


def audio():
    global _audio
    if _audio is None:
        _audio = Audio(os.path.join(BUILD, "features.npz"))
    return _audio


def circle_path(x, y, r):
    p = skia.Path()
    p.addCircle(x, y, r)
    return p


def rrect(l, t, r, b, rad):
    p = skia.Path()
    p.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(l, t, r, b), rad, rad))
    return p


def bez_q(p0, p1, p2, n=12):
    out = []
    for i in range(n + 1):
        u = i / n
        out.append(((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0],
                    (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1]))
    return out


def path_from(pts, close=True, smooth_k=0.0):
    """Polyline or (smooth_k>0) Catmull-Rom smoothed closed/open path."""
    path = skia.Path()
    n = len(pts)
    if n == 0:
        return path
    if smooth_k <= 0 or n < 3:
        path.moveTo(*pts[0])
        for p in pts[1:]:
            path.lineTo(*p)
        if close:
            path.close()
        return path
    path.moveTo(*pts[0])
    rng = range(n) if close else range(n - 1)
    for i in rng:
        p0 = pts[(i - 1) % n] if (close or i > 0) else pts[i]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (close or i + 2 < n) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) * smooth_k / 3, p1[1] + (p2[1] - p0[1]) * smooth_k / 3)
        c2 = (p2[0] - (p3[0] - p1[0]) * smooth_k / 3, p2[1] - (p3[1] - p1[1]) * smooth_k / 3)
        path.cubicTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    if close:
        path.close()
    return path
