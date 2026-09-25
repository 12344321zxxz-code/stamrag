"""Shared scene context + text/graphic helpers used by all CHASE scenes."""
import math

import numpy as np
import skia

from . import fxk as FX
from . import hero as Z
from .core import (BM, H, TAU, W, C, P, audio, clamp, ease_back, ease_in, ease_io, ease_out, font, fract, hash1,
                   inv_lerp, lerp, lin_grad, mix, path_from, poly, pulse, rad_grad, scale_c, smooth, vnoise)

A = audio()

# bar ranges of each section (bar 0 = the band's first downbeat at 24.2 s)
SECTIONS = [("cold", -99, 0), ("title", 0, 1), ("roof", 1, 9), ("train", 9, 17), ("build", 17, 21),
            ("chorus1", 21, 28), ("panels", 28, 32), ("solo", 32, 40), ("inside", 40, 45), ("final", 45, 59),
            ("victory", 59, 63), ("end", 63, 999)]


def bar_t(bar):
    if bar <= -99:
        return -1.0
    if bar >= 999:
        return A.duration + 1
    return A.bar_time(bar)


class Ctx:
    def __init__(self, t, frame):
        self.t = t
        self.frame = frame
        self.b = A.beat(t)
        self.bi = math.floor(self.b)
        self.u = self.b - self.bi
        self.bar = self.b / 4.0
        for k in ("kick", "snare", "hat", "rms", "bass", "gtr", "onset", "speech"):
            setattr(self, k, A.e(k, t))
        for name, b0, b1 in SECTIONS:
            if bar_t(b0) <= t < bar_t(b1):
                self.sec, self.bar0, self.bar1 = name, b0, b1
                break
        else:
            self.sec, self.bar0, self.bar1 = SECTIONS[-1]
        self.s0 = max(0.0, bar_t(self.bar0))
        self.lt = t - self.s0
        self.lb = self.b - self.bar0 * 4 if self.bar0 > -99 else self.b
        self.lbar = self.lb / 4.0

    def at_bar(self, bar):
        return A.bar_time(bar)

    def since(self, t0):
        return self.t - t0

    def beat_age(self, beat):
        return self.t - A.beat_time(beat)

    def shot(self, cuts):
        """cuts are absolute bar numbers; returns (index, beats since that cut)."""
        idx = 0
        for i, cut in enumerate(cuts):
            if self.bar >= cut:
                idx = i
        return idx, self.b - cuts[idx] * 4


def post(**kw):
    d = dict(bloom=0.35, threshold=0.6, grain=0.06, vignette=0.45, chroma=0.0, flash=0.0, flash_col="#ffffff",
             sat=1.08, contrast=1.05, bright=0.0, tint=(1, 1, 1), letterbox=0.0, scan=0.0, fade=0.0, impact=0.0,
             glitch=0.0, zoom_blur=0.0, zb_center=(W / 2, H / 2), smear=(0.0, 0.0), invert=True)
    d.update(kw)
    return d


def cam(c, zoom=1.0, cx=W / 2, cy=H / 2, rot=0.0, ox=0.0, oy=0.0):
    c.save()
    c.translate(W / 2 + ox, H / 2 + oy)
    c.rotate(rot)
    c.scale(zoom, zoom)
    c.translate(-cx, -cy)


def shake(x, amt, seed=1):
    return vnoise(x.t * 37.0, seed) * amt, vnoise(x.t * 41.0, seed + 1) * amt


def punch(x, amt=0.03, k=7.0):
    return 1 + amt * math.exp(-x.u * k) * (0.4 + 0.6 * x.kick)


def zdraw(c, pose, x, y, size, t, mirror=False, **kw):
    pose["t"] = t
    if pose.get("guitar") == "play" and pose["hands"][1] is None:
        Z.finish_play_hands(pose)
    Z.draw_zip(c, pose, x, y, size, mirror=mirror, **kw)


def note_age(t, min_pitch=0, lead=True):
    n = A.last_note(t, lead=lead, min_pitch=min_pitch)
    return (t - n[0]) if n is not None else 9.0


def recent_notes(t, life=0.6, min_pitch=0, lead=True):
    arr = A.notes_in(t - life, t + 1e-6, lead=lead)
    return [n for n in arr if n[2] >= min_pitch]


# ----------------------------------------------------------------------------
# speech bubbles
# ----------------------------------------------------------------------------
def bubble(c, text, x, y, tail, age, progress=1.0, size=64, fnt="PatrickHand.ttf", fill="#ffffff", ink="#140c1c",
           maxw=760, shout=False, a=1.0, text_col=None, highlight=None):
    """Speech balloon centred at (x,y) with its tail pointing to `tail`. Text types on with `progress`."""
    if age < 0:
        return
    k = ease_back(clamp(age / 0.18), 2.0)
    f = font(fnt, size)
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if f.measureText(trial) > maxw and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    lw = max(f.measureText(l) for l in lines)
    lh = size * 1.1
    bw, bh = lw + size * 1.2, lh * len(lines) + size * 0.7
    c.save()
    c.translate(x, y)
    c.scale(k, k)
    c.translate(-x, -y)
    # tail
    tx, ty = tail
    ang = math.atan2(ty - y, tx - x)
    nx, ny = -math.sin(ang), math.cos(ang)
    base = size * 0.35
    tp = poly([(x + nx * base, y + ny * base), (tx, ty), (x - nx * base, y - ny * base)])
    if shout:
        pts = []
        n = 22
        for i in range(n):
            u = i / n * TAU
            rr = 1.0 if i % 2 == 0 else 0.82
            pts.append((x + math.cos(u) * bw * 0.62 * rr, y + math.sin(u) * bh * 0.7 * rr))
        body = poly(pts)
    else:
        body = skia.Path()
        body.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2),
                                            bh * 0.45, bh * 0.45))
    c.drawPath(tp, P(ink, a, stroke=size * 0.14))
    c.drawPath(body, P(ink, a, stroke=size * 0.14))
    c.drawPath(tp, P(fill, a))
    c.drawPath(body, P(fill, a))
    # text (typewriter by characters)
    total = sum(len(l) for l in lines)
    shown = int(round(total * clamp(progress)))
    yy = y - bh / 2 + size * 0.35 + size * 0.85
    for l in lines:
        s = l[:max(0, shown)]
        shown -= len(l)
        lwid = f.measureText(l)
        if s:
            if highlight and highlight in l:
                # draw the highlighted word separately so it can wobble/colour
                pre, _, post_ = l.partition(highlight)
                xx = x - lwid / 2
                c.drawString(pre[:len(s)], xx, yy, f, P(text_col or ink, a))
            else:
                c.drawString(s, x - lwid / 2, yy, f, P(text_col or ink, a))
        yy += lh
    c.restore()
    return (x - bw / 2 * k, y - bh / 2 * k, x + bw / 2 * k, y + bh / 2 * k)


# ----------------------------------------------------------------------------
# lyric renderers (original text, word timings from lyrics.Line.state)
# ----------------------------------------------------------------------------
def layout_words(words, f, maxw):
    lines, cur, cw = [], [], 0.0
    sp = f.measureText(" ")
    for i, w in enumerate(words):
        ww = f.measureText(w)
        if cur and cw + sp + ww > maxw:
            lines.append((cur, cw))
            cur, cw = [], 0.0
        cur.append((i, w, ww))
        cw = cw + (sp if len(cur) > 1 else 0) + ww
    if cur:
        lines.append((cur, cw))
    return lines, sp


PALETTES = {
    "hot": ("#ffd21f", "#ff3d8a", "#140a1c"),
    "cool": ("#7afcff", "#2fe3d7", "#0a0a24"),
    "pink": ("#ffffff", "#ff5fa8", "#3a0a24"),
    "gold": ("#fff4c0", "#ffb31a", "#2a1400"),
}


def slam_text(c, line, t, cx, cy, size=120, fnt="DelaGothicOne.ttf", pal="hot", maxw=1500, a_mul=1.0, tilt=-4.0,
              shake_amt=6.0, beat_u=0.0, kick=0.0, stroke_k=0.16, align_y="center"):
    a, words = line.state(t)
    if a <= 0 or not words:
        return
    f = font(fnt, size)
    lines, sp = layout_words([w for w, _, _ in words], f, maxw)
    lh = size * 1.18
    top = cy - lh * (len(lines) - 1) / 2 if align_y == "center" else cy
    fill, accent, ink = PALETTES[pal]
    c.save()
    c.translate(cx, cy)
    c.rotate(tilt)
    c.translate(-cx, -cy)
    for li, (items, lw_) in enumerate(lines):
        x = cx - lw_ / 2
        y = top + li * lh + size * 0.35
        for (i, w, ww) in items:
            _, age, cur = words[i]
            if age >= -0.02:
                k = ease_back(clamp(age / 0.16), 3.0)
                j = shake_amt * (1 - clamp(age / 0.3))
                jx, jy = math.sin(age * 70 + i) * j, math.cos(age * 63 + i) * j
                c.save()
                c.translate(x + ww / 2 + jx, y - size * 0.35 + jy)
                sc = k * (1.08 if cur else 1.0) * (1 + 0.04 * kick * (1 if cur else 0))
                c.scale(sc, sc)
                col_ = mix(accent, fill, 0.0) if cur else fill
                FX.outlined_text(c, w, 0, size * 0.35, size, fnt, fill=col_, stroke=ink, sw=size * stroke_k, a=a * a_mul,
                                 extra_stroke=accent if cur else None, shadow=True)
                c.restore()
            x += ww + sp
    c.restore()


def mega_text(c, line, t, cx, cy, size=240, fnt="RubikMonoOne.ttf", pal="hot", a_mul=1.0, kick=0.0):
    """Huge chant words: each one slams full-screen, previous ones shrink to the background."""
    a, words = line.state(t, fade_out=0.25)
    if a <= 0:
        return
    fill, accent, ink = PALETTES[pal]
    shown = [(i, w, age) for i, (w, age, cur) in enumerate(words) if age >= 0]
    for idx, (i, w, age) in enumerate(shown):
        latest = idx == len(shown) - 1
        k = ease_back(clamp(age / 0.14), 3.5)
        s = size * (1.0 if latest else 0.55)
        y = cy + (0 if latest else -size * 0.95 + (len(shown) - 2 - idx) * -s * 0.9)
        f = font(fnt, s)
        c.save()
        c.translate(cx, y)
        c.rotate(-6 + (i % 2) * 4)
        c.scale(k * (1 + 0.05 * kick), k * (1 + 0.05 * kick))
        if latest:
            FX.outlined_text(c, w, 0, s * 0.36, s, fnt, fill=fill, stroke=ink, sw=s * 0.14, a=a * a_mul,
                             extra_stroke=accent, glow=accent)
        else:
            FX.outlined_text(c, w, 0, s * 0.36, s, fnt, fill=accent, stroke=ink, sw=s * 0.12, a=0.8 * a * a_mul)
        c.restore()


def fly_text(c, line, t, vp=(W / 2, H * 0.42), size=150, fnt="BlackOpsOne.ttf", pal="cool", travel=1.4):
    """Words launch from the vanishing point and rush past the camera."""
    a, words = line.state(t, fade_out=0.6)
    if not words:
        return
    fill, accent, ink = PALETTES[pal]
    for i, (w, age, cur) in enumerate(words):
        if age < 0 or age > travel:
            continue
        f = age / travel
        z = lerp(0.1, 2.6, f ** 1.6)
        side = (-1 if i % 2 == 0 else 1)
        x = vp[0] + side * (80 + 520 * f ** 1.5) * (0.6 + 0.4 * hash1(i))
        y = vp[1] + (-1 if (i // 2) % 2 == 0 else 1) * 180 * f ** 1.5
        al = clamp(f * 6) * clamp((1 - f) * 5)
        c.save()
        c.translate(x, y)
        c.scale(z, z)
        FX.outlined_text(c, w, 0, size * 0.35, size, fnt, fill=fill if not cur else "#ffffff", stroke=ink, sw=size * 0.12,
                         a=al, extra_stroke=accent if cur else None)
        c.restore()


def float_text(c, line, t, cx, cy, size=86, fnt="PatrickHand.ttf", color="#ffffff", ink="#7a1048", maxw=1300):
    a, words = line.state(t, fade_in=0.4, fade_out=0.5)
    if a <= 0:
        return
    f = font(fnt, size)
    lines, sp = layout_words([w for w, _, _ in words], f, maxw)
    lh = size * 1.15
    top = cy - lh * (len(lines) - 1) / 2
    for li, (items, lw_) in enumerate(lines):
        x = cx - lw_ / 2
        y = top + li * lh
        for (i, w, ww) in items:
            _, age, cur = words[i]
            if age >= 0:
                k = ease_out(clamp(age / 0.35))
                bob = math.sin(t * 2 + i * 0.7) * 6
                c.drawString(w, x, y + bob + (1 - k) * 30, f, P(ink, a * k * 0.6, stroke=size * 0.12))
                c.drawString(w, x, y + bob + (1 - k) * 30, f, P("#fff2a0" if cur else color, a * k))
            x += ww + sp


def sky_text(c, line, t, cx, cy, size=150, fnt="Bangers.ttf", color="#fff4c0", glow="#ffb31a"):
    a, words = line.state(t, fade_out=0.6)
    if a <= 0:
        return
    f = font(fnt, size)
    lines, sp = layout_words([w for w, _, _ in words], f, 1700)
    for li, (items, lw_) in enumerate(lines):
        x = cx - lw_ / 2
        y = cy + li * size * 1.1
        for (i, w, ww) in items:
            _, age, cur = words[i]
            if age >= 0:
                k = ease_back(clamp(age / 0.25), 2.0)
                c.save()
                c.translate(x + ww / 2, y)
                c.scale(k, k)
                c.drawString(w, -ww / 2, 0, f, P(glow, 0.7 * a, blur=size * 0.15))
                c.drawString(w, -ww / 2, 0, f, P("#2a0a18", a, stroke=size * 0.1))
                c.drawString(w, -ww / 2, 0, f, P(color, a))
                c.restore()
            x += ww + sp


# ----------------------------------------------------------------------------
# melody ribbon: the guitar line drawn as a glowing trail
# ----------------------------------------------------------------------------
def melody_ribbon(c, t, x_now, speed, y_of, back=4.0, fwd=0.0, color="#7afcff", width=8.0, a=1.0, heads=True,
                  min_pitch=50):
    notes = A.notes_in(t - back, t + fwd, lead=True)
    notes = [n for n in notes if n[2] >= min_pitch]
    if len(notes) < 2:
        return
    pts = []
    for s, e, p, amp in notes:
        x0 = x_now + (s - t) * speed
        x1 = x_now + (min(e, t) - t) * speed
        y = y_of(p)
        pts.append((x0, y))
        if x1 > x0 + 4:
            pts.append((x1, y))
    pts.append((x_now, y_of(notes[-1][2])))
    path = path_from(pts, close=False, smooth_k=0.5)
    c.drawPath(path, P(color, 0.35 * a, stroke=width * 4, blur=width * 2))
    c.drawPath(path, P(color, 0.9 * a, stroke=width))
    c.drawPath(path, P("#ffffff", a, stroke=width * 0.35))
    if heads:
        for s, e, p, amp in notes:
            age = t - s
            x0 = x_now + (s - t) * speed
            r = width * (1.2 + 1.5 * clamp(1 - age * 3))
            c.drawCircle(x0, y_of(p), r, P("#ffffff", a))
            c.drawCircle(x0, y_of(p), r * 2.5, P(color, 0.4 * a * clamp(1 - age * 2), blur=r))
