"""The timeline: one draw function per song section, all driven by the beat grid.

Every scene returns a dict of post-processing settings consumed by render.py.
"""
import math

import numpy as np
import skia

from . import fx
from . import props as pr
from . import rig
from .core import (BM, H, P, TAU, W, audio, clamp, col, draw_text_c, ease_back, ease_in, ease_io, ease_out, fbm,
                   font, fract, hash1, hit, inv_lerp, lerp, lin_grad, mix, poly, pulse, rad_grad, remap, scale_c,
                   smooth, vnoise)

A = audio()

# (name, start_bar, end_bar) -- see README for how these were derived from the audio
SECTIONS = [
    ("intro", -1, 2), ("verse1", 2, 15), ("hook1", 15, 18), ("break1", 18, 20), ("verse2", 20, 33),
    ("hook2", 33, 36), ("break2", 36, 38), ("neon", 38, 50), ("bridge", 50, 54), ("dawn", 54, 67),
    ("hook3", 67, 70), ("hall", 70, 73), ("stop", 73, 74), ("finale", 74, 81), ("outro", 81, 999),
]


def sec_time(bar):
    if bar < 0:
        return 0.0
    if bar >= 999:
        return A.duration + 0.5
    return A.bar_time(bar)


class Ctx:
    def __init__(self, t, frame):
        self.t = t
        self.frame = frame
        self.b = A.beat(t)
        self.bi = math.floor(self.b)
        self.u = self.b - self.bi
        self.kick = A.e("kick", t)
        self.snare = A.e("snare", t)
        self.hat = A.e("hat", t)
        self.rms = A.e("rms", t)
        self.low = A.e("low", t)
        self.period = A.period
        for name, b0, b1 in SECTIONS:
            if sec_time(b0) <= t < sec_time(b1):
                self.sec, self.bar0, self.bar1 = name, b0, b1
                break
        else:
            self.sec, self.bar0, self.bar1 = SECTIONS[-1]
        self.s0 = sec_time(self.bar0)
        self.s1 = min(sec_time(self.bar1), A.duration + 0.5)
        self.lt = t - self.s0
        self.lb = self.b - max(self.bar0, 0) * 4
        self.bar = self.b / 4.0
        self.lbar = self.lb / 4.0

    def shot(self, bars):
        """Given local bar cut points [0, 4, 8, ...], return (index, local beat within shot)."""
        idx = 0
        for i, cut in enumerate(bars):
            if self.lbar >= cut:
                idx = i
        return idx, self.lb - bars[idx] * 4


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------
def cam(c, zoom=1.0, cx=W / 2, cy=H / 2, rot=0.0, ox=0.0, oy=0.0):
    c.save()
    c.translate(W / 2 + ox, H / 2 + oy)
    c.rotate(rot)
    c.scale(zoom, zoom)
    c.translate(-cx, -cy)


def shake(x, amt):
    return (vnoise(x.t * 37.0, 1) * amt, vnoise(x.t * 41.0, 2) * amt)


PAINT_STYLE = rig.Style("paint", ink="#5a2412", ink_w=0.008, shade=False)
SIL = rig.Style("silhouette")


def dancer(c, x, name, style, b, px, gy, size, facing=1, draw_style=rig.FLAT, i=0, stride=0.17, energy=1.0,
           sil_color="black", scroll_adv=None):
    pose, adv = rig.dance(style, b, i, stride=stride, energy=energy)
    if scroll_adv is not None:
        adv -= scroll_adv
    rig.draw_dancer(c, name, pose, px + adv * size * facing, gy, size, facing, draw_style, sil_color, x.t)
    return pose, adv


def long_shadow(c, x, name, style, b, px, gy, size, facing=1, k=1.4, sy=0.22, a=0.28, i=0, stride=0.17,
                scroll_adv=None, color="#2b140a"):
    pose, adv = rig.dance(style, b, i, stride=stride)
    if scroll_adv is not None:
        adv -= scroll_adv
    X = px + adv * size * facing
    c.saveLayerAlpha(skia.Rect.MakeLTRB(X - size * 3, gy - size * 0.2, X + size * 1.2, gy + size * 0.6), int(a * 255))
    c.save()
    c.translate(X, gy)
    c.concat(skia.Matrix.MakeAll(1, k, 0, 0, -sy, 0, 0, 0, 1))
    rig.draw_dancer(c, name, pose, 0, 0, size, facing, SIL, color, x.t)
    c.restore()
    c.restore()


def ground_shadow(c, px, gy, size, a=0.3):
    c.drawOval(skia.Rect.MakeLTRB(px - size * 0.2, gy - size * 0.025, px + size * 0.24, gy + size * 0.025),
               P("#000000", a, blur=size * 0.02))


def landing_puffs(c, x, px0, gy, size, b, stride=0.17, scroll_adv=0.0, color="#e8c58a", a=0.8):
    """Dust puff where the swinging foot lands each beat (fixed in world space)."""
    k = math.floor(b)
    u = b - k
    land = 0.62
    for kk, age_b in ((k, u - land), (k - 1, u + 1 - land)):
        if age_b < 0:
            continue
        plant = stride * (kk + 1) + stride * 0.5
        X = px0 + (plant - scroll_adv) * size
        pr.dust_puff(c, X, gy, age_b * A.period, s=size / 300, color=color, a=a)


_img_cache = {}


def cached_img(key, fn):
    im = _img_cache.get(key)
    if im is None:
        im = fn()
        _img_cache[key] = im
    return im


def sandstone(w, h, base=(0.86, 0.7, 0.49), seed=0):
    rng = np.random.default_rng(seed)
    small = rng.normal(0, 1, (h // 16 + 2, w // 16 + 2))
    from scipy.ndimage import gaussian_filter, zoom
    lo = gaussian_filter(small, 1.5)
    lo = zoom(lo, 16, order=1)[:h, :w]
    hi = rng.normal(0, 1, (h, w))
    hi = gaussian_filter(hi, 0.8)
    v = 1 + lo * 0.06 + hi * 0.035
    speck = rng.random((h, w)) < 0.002
    v[speck] *= 0.75
    img = np.zeros((h, w, 4), np.uint8)
    for i in range(3):
        img[..., i] = np.clip(base[i] * v * 255, 0, 255)
    img[..., 3] = 255
    return skia.Image.fromarray(img, colorType=skia.kRGBA_8888_ColorType)


def title(c, cx, cy, reveal, s=1.0, a=1.0, shine=0.0, glow_a=0.0):
    """Gold title. reveal: 4 floats in [0,1] for the four words."""
    f1 = font("CinzelDecorative-Black.ttf", 118 * s)
    f2 = font("CinzelDecorative-Black.ttf", 196 * s)
    words1 = ["WALK", "LIKE", "AN"]
    sp = 38 * s
    ws = [f1.measureText(w) for w in words1]
    total = sum(ws) + sp * 2
    x = cx - total / 2
    y1 = cy - 40 * s
    y2 = cy + 170 * s
    items = []
    for i, w in enumerate(words1):
        items.append((w, x + ws[i] / 2, y1, f1, reveal[i]))
        x += ws[i] + sp
    items.append(("EGYPTIAN", cx, y2, f2, reveal[3]))
    gold = lin_grad((0, cy - 160 * s), (0, cy + 190 * s),
                    [(0, "#fff6c8", a), (0.35, "#f2c14e", a), (0.55, "#b8741a", a), (0.62, "#ffe08a", a), (1, "#c98a1e", a)])
    for (w, wx, wy, f, r) in items:
        if r <= 0:
            continue
        k = ease_back(r, 2.2)
        al = clamp(r * 3) * a
        c.save()
        c.translate(wx, wy - f.getSize() * 0.35)
        c.scale(k, k)
        c.translate(-wx, -(wy - f.getSize() * 0.35))
        tw = f.measureText(w)
        if glow_a > 0:
            c.drawString(w, wx - tw / 2, wy, f, P("#ffb03a", glow_a * al, blur=18 * s))
        c.drawString(w, wx - tw / 2 + 6 * s, wy + 8 * s, f, P("#1a0800", 0.6 * al, blur=6 * s))
        c.drawString(w, wx - tw / 2, wy, f, P("#3a1c05", al, stroke=10 * s))
        gp = skia.Paint(AntiAlias=True, Shader=gold)
        gp.setAlphaf(al)
        c.drawString(w, wx - tw / 2, wy, f, gp)
        if shine > 0:
            sx = lerp(cx - 900 * s, cx + 900 * s, shine)
            sh = lin_grad((sx - 90 * s, 0), (sx + 90 * s, 0), [(0, "#ffffff", 0), (0.5, "#ffffff", 0.85 * al), (1, "#ffffff", 0)])
            c.drawString(w, wx - tw / 2, wy, f, P(shader=sh))
        c.restore()
    # decorative bars
    bar_a = clamp(reveal[3] * 2) * a
    if bar_a > 0:
        half = 720 * s * ease_out(reveal[3])
        for yy in (cy - 190 * s, cy + 225 * s):
            c.drawLine(cx - half, yy, cx + half, yy, P("#f2c14e", bar_a, stroke=5 * s))
            c.drawLine(cx - half, yy + 12 * s, cx + half, yy + 12 * s, P("#1f3f95", bar_a, stroke=4 * s))
        pr.eye_of_horus(c, cx - half - 70 * s, cy + 20 * s, 34 * s, color="#f2c14e", a=bar_a, fill_iris=False)
        c.save()
        c.translate(cx + half + 70 * s, cy + 20 * s)
        c.scale(-1, 1)
        pr.eye_of_horus(c, 0, 0, 34 * s, color="#f2c14e", a=bar_a, fill_iris=False)
        c.restore()


def default_post(**kw):
    d = dict(bloom=0.35, threshold=0.6, grain=0.07, vignette=0.5, chroma=0.0, flash=0.0, flash_col="#ffffff",
             sat=1.08, contrast=1.05, bright=0.0, tint=(1, 1, 1), letterbox=0.0, scan=0.0, fade=0.0, iris=1.0)
    d.update(kw)
    return d


def cut_flash(x, dur=0.35, peak=0.85):
    return peak * clamp(1 - x.lt / dur) ** 2


# ----------------------------------------------------------------------------
# INTRO: sunrise + title
# ----------------------------------------------------------------------------
def s_intro(c, x):
    t = x.t
    T = x.s1
    p = smooth(t / T)
    zoom_k = ease_in(inv_lerp(A.beat_time(6.5), T, t), 3)
    sun_y = lerp(930, 600, ease_out(clamp(t / 4.2), 2))
    sun_x = W / 2 - 68
    zc = min(1, zoom_k * 3)
    cam(c, zoom=1 + zoom_k * 7, cx=lerp(W / 2, sun_x, zc), cy=lerp(H / 2, sun_y, zc))
    pr.sky(c, [(0, mix("#02030c", "#1a0f3a", p)), (0.45, mix("#070b22", "#6a2a5a", p)),
               (0.7, mix("#0d1030", "#ff7a3a", p)), (1, mix("#141236", "#ffd08a", p))], 0, 780)
    pr.stars(c, t, alpha=clamp(t / 0.8) * (1 - p * 0.8), seed=3, n=500, h=700)
    pr.sun(c, sun_x, sun_y, 175, glow_a=0.35 + 0.4 * p + 0.25 * x.kick, rays=p, ray_t=t)
    if p > 0.5:
        pr.aten_rays(c, sun_x, sun_y, 175, t, a=(p - 0.5) * 2 * 0.8)
    # pyramid silhouettes with rim light
    for (px, w, h) in ((W / 2 + 330, 820, 400), (W / 2 - 360, 600, 300), (W / 2 - 720, 330, 165), (W / 2 + 820, 300, 140)):
        pr.pyramid(c, px, 790, w, h, lit="#2a1420", shade="#150a14", light=-1, courses=False)
        c.drawLine(px, 790 - h, px + w / 2, 790, P("#ffb070", 0.55 * p, stroke=3))
    pr.dune(c, t * 8, 800, 16, 2.0, "#1d0f18", "#0b0509")
    pr.dune(c, t * 16, 900, 22, 5.0, "#140a10", "#050204")
    # hat-driven glints on the big pyramid edge
    for i in range(6):
        f = hash1(i * 7 + int(x.b * 4))
        gx, gy = lerp(W / 2 + 330, W / 2 + 330 + 410, f), lerp(390, 790, f)
        pr.glow(c, gx, gy, 30 + 50 * x.hat, "#ffe6b0", 0.6 * x.hat * p)
    c.restore()
    # title words land on beats 2..5
    rev = [clamp((x.b - (2 + i)) / 0.35) for i in range(4)]
    shine = clamp((x.b - 5.5) / 1.5)
    ta = 1 - zoom_k
    if ta > 0:
        title(c, W / 2, 250, rev, s=0.95, a=ta, shine=shine, glow_a=0.6 * rev[3])
    fl = 0.0
    for i in range(4):
        fl = max(fl, 0.35 * pulse(x.b - (2 + i), 7) if x.b >= 2 + i else 0)
    fl = max(fl, ease_in(zoom_k, 2))
    return default_post(bloom=0.55, threshold=0.5, flash=fl, flash_col="#fff2d0", vignette=0.6,
                        fade=clamp(1 - t / 0.6))


# ----------------------------------------------------------------------------
# VERSE 1: golden desert procession
# ----------------------------------------------------------------------------
CAST1 = ["nefi", "khet", "anpu", "miu", "heru", "djehu"]


def desert_bg(c, x, scroll, sun_pos=(1330, 560), sun_r=150, horizon=640, pyr=True, sky_k=0.0):
    t = x.t
    pr.sky(c, [(0, mix("#2b1d5e", "#1c1240", sky_k)), (0.38, "#9b3f63"), (0.68, "#f08a4b"), (0.9, "#ffd08a"),
               (1, "#fff0c0")], 0, horizon + 10)
    pr.stars(c, t, alpha=0.25, seed=5, n=200, h=300)
    pr.sun(c, sun_pos[0], sun_pos[1], sun_r, glow_a=0.55 + 0.25 * x.kick, rays=0.4, ray_t=t)
    for i, (cx, cy, w) in enumerate(((300, 180, 380), (1100, 120, 460), (1650, 250, 300))):
        pr.cloud(c, (cx - scroll * 0.03 + i * 50) % (W + 600) - 300, cy, w, 60, "#ffc3a0", 0.22, seed=i)
    if pyr:
        base = horizon + 6
        off = -scroll * 0.08
        for (px, w, h) in ((380, 520, 250), (760, 360, 170), (1600, 640, 300), (1180, 250, 110)):
            X = (px + off) % (W + 900) - 450
            pr.pyramid(c, X, base, w, h, lit="#f6b565", shade="#a8562e", light=-1, cap=0.3)
    pr.dune(c, scroll * 0.15, horizon + 12, 14, 1.3, "#e39a5a", "#c87444", rim="#ffd9a0")
    pr.dune(c, scroll * 0.35, horizon + 110, 26, 4.1, "#eeb070", "#cc8248", rim="#ffe2b0")


def camels(c, x, scroll, y, s=38, n=5, a=0.9):
    for i in range(n):
        X = (i * 90 + x.t * 25 - scroll * 0.35) % (W + 400) - 200
        pr.camel(c, X, y + math.sin(X * 0.004) * 6, s, x.t + i * 0.7, color="#5a2a24", a=a)


def birds(c, x, n=5, y0=200, seed=0, color="#3a1a20", a=0.8):
    for i in range(n):
        X = (i * 170 + x.t * 60 + seed * 200) % (W + 300) - 150
        Y = y0 + math.sin(i * 1.7 + x.t * 0.5) * 30 + i * 12
        f = math.sin(x.t * 8 + i) * 8
        pth = skia.Path()
        pth.moveTo(X - 14, Y - f)
        pth.quadTo(X - 6, Y - 4, X, Y)
        pth.quadTo(X + 6, Y - 4, X + 14, Y - f)
        c.drawPath(pth, P(color, a, stroke=2.5))


def s_verse1(c, x):
    idx, sb = x.shot([0, 4, 8, 13])
    b = x.b
    post = default_post(bloom=0.3, flash=cut_flash(x, 0.5, 0.9) if x.lt < 0.5 else 0.0, flash_col="#fff0c0")
    if idx == 0:
        size, stride, gy = 270, 0.17, 930
        cam_adv = stride * b * 0.86
        scroll = cam_adv * size
        cam(c, zoom=1.0 + 0.02 * pulse(x.u, 6) * x.kick)
        desert_bg(c, x, scroll)
        camels(c, x, scroll, 745)
        birds(c, x, 4, 180)
        # near ground band with pebbles scrolling in lock with the walkers
        c.drawRect(skia.Rect.MakeLTRB(0, gy - 70, W, H), P(shader=lin_grad((0, gy - 70), (0, H), [(0, "#f2c07e", 1), (1, "#d18d52", 1)])))
        for i in range(60):
            px = (i * 97.3 - scroll) % (W + 100) - 50
            py = gy - 50 + (i * 37 % 150)
            c.drawOval(skia.Rect.MakeLTRB(px, py, px + 10 + i % 7, py + 4), P("#b8743e", 0.5))
        x0 = 240
        for i, n in enumerate(CAST1):
            px = x0 + i * 250
            long_shadow(c, x, n, "egypt_walk", b - i * 0.0, px, gy, size, k=1.5, i=i, scroll_adv=cam_adv)
        for i, n in reversed(list(enumerate(CAST1))):
            px = x0 + i * 250
            landing_puffs(c, x, px, gy, size, b, scroll_adv=cam_adv, color="#f5d5a0")
            dancer(c, x, n, "egypt_walk", b, px, gy, size, i=i, scroll_adv=cam_adv)
        c.restore()
    elif idx == 1:
        size, stride, gy = 600, 0.17, 1045
        cam_adv = stride * b
        scroll = cam_adv * size
        zoom = 1.0 + 0.03 * (sb / 16) + 0.025 * pulse(x.u, 7) * (1 if x.bi % 2 == 0 else 0.4)
        cam(c, zoom=zoom, cy=H * 0.55)
        desert_bg(c, x, scroll * 0.6, sun_pos=(1500, 420), sun_r=190, horizon=700)
        c.drawRect(skia.Rect.MakeLTRB(-200, gy - 60, W + 200, H + 200), P(shader=lin_grad((0, gy - 60), (0, H), [(0, "#f2c07e", 1), (1, "#c98446", 1)])))
        for i in range(40):
            px = (i * 131.3 - scroll) % (W + 300) - 150
            c.drawOval(skia.Rect.MakeLTRB(px, gy - 30 + i * 13 % 70, px + 22, gy - 22 + i * 13 % 70), P("#b8743e", 0.45))
        for i, n in enumerate(["nefi", "khet"]):
            px = 620 + i * 560
            landing_puffs(c, x, px, gy, size, b, scroll_adv=cam_adv, color="#f5d5a0")
            ground_shadow(c, px, gy, size, 0.25)
            dancer(c, x, n, "egypt_walk", b, px, gy, size, i=i, scroll_adv=cam_adv, energy=1.35)
        # foreground palm silhouettes whooshing past
        for j in range(2):
            X = (j * 1400 - scroll * 1.8) % (W + 1600) - 500
            pr.palm(c, X, H + 60, 950, x.t, lean=0.05, sil=True, color="#1a0c0a", seed=j)
        c.restore()
        post.update(bloom=0.35)
    else:
        size, stride, gy = 300, 0.17, 850
        cam_adv = stride * b * 0.9
        scroll = cam_adv * size
        last = clamp((x.lbar - 12) / 1.0)
        zoom = 1.0 + ease_in(last, 3) * 2.2
        cam(c, zoom=zoom, cx=W / 2 + 120 * ease_in(last), cy=720)
        pr.sky(c, [(0, "#3a0f2a"), (0.45, "#b8323a"), (0.8, "#ff8a3a"), (1, "#ffd07a")], 0, 860)
        pr.sun(c, W / 2, 700, 430, core="#fff0b0", edge="#ff6a1f", glow_c="#ff8a3a", glow_a=0.55 + 0.35 * x.kick)
        # heat haze stripes across the sun
        for i in range(7):
            yy = 560 + i * 45 + math.sin(x.t * 2 + i) * 6
            c.drawRect(skia.Rect.MakeLTRB(W / 2 - 440, yy, W / 2 + 440, yy + 3 + i), P("#ff7a2a", 0.25))
        birds(c, x, 6, 260, seed=2, color="#2a0a10")
        pr.dune(c, scroll * 0.3, 870, 10, 3.3, "#1a0a0c", "#0b0406")
        c.drawRect(skia.Rect.MakeLTRB(-400, gy, W + 400, H + 300), P("#120607"))
        for i, n in enumerate(CAST1 + ["tut"]):
            px = 180 + i * 240
            dancer(c, x, n, "egypt_walk", b - i * 0.08, px, gy, size, i=i, draw_style=SIL, sil_color="#120607",
                   scroll_adv=cam_adv)
        pr.camel(c, (x.t * 30) % (W + 400) - 200, gy, 70, x.t, color="#120607")
        c.restore()
        post.update(bloom=0.45, threshold=0.5, flash=ease_in(last, 4) * 0.95, flash_col="#fff6e0",
                    chroma=4 * ease_in(last, 2))
    return post


# ----------------------------------------------------------------------------
# HOOK: tomb wall comes alive
# ----------------------------------------------------------------------------
def frieze(c, x0, y, w, h, t=0.0, lit=0.0):
    cols = ["#1f3f95", "#c0392b", "#f2c14e", "#2f7d4a"]
    n = int(w / 36)
    for i in range(n):
        cc = cols[i % 4]
        if lit > 0 and (i + int(t * 8)) % 8 == 0:
            cc = mix(cc, "#fff4c0", lit)
        c.drawRect(skia.Rect.MakeLTRB(x0 + i * 36 + 3, y, x0 + i * 36 + 33, y + h), P(cc))
        c.drawRect(skia.Rect.MakeLTRB(x0 + i * 36 + 12, y + h * 0.18, x0 + i * 36 + 24, y + h * 0.72), P("#f3e9d2", 0.85))


def wall_scene(c, x, rows=3, style="hook", sat_boost=0.0, palette_t=0.0):
    wall = cached_img("wall", lambda: sandstone(W + 400, H + 400, (0.86, 0.69, 0.47), 11))
    c.drawImage(wall, -200, -200)
    reg_h = 300
    top = 90
    cast = [["nefi", "khet", "anpu", "miu", "heru", "djehu", "meri"],
            ["tut", "sobi", "miu", "nefi", "anpu", "khet", "heru"],
            ["djehu", "heru", "meri", "tut", "sobi", "nefi", "anpu"]]
    frieze(c, -200, 20, W + 400, 58, x.t, lit=x.kick)
    for r in range(rows):
        y0 = top + r * (reg_h + 22)
        c.drawRect(skia.Rect.MakeLTRB(-200, y0 + reg_h, W + 200, y0 + reg_h + 22), P("#1f3f95"))
        c.drawRect(skia.Rect.MakeLTRB(-200, y0 + reg_h + 6, W + 200, y0 + reg_h + 12), P("#f2c14e"))
        c.drawRect(skia.Rect.MakeLTRB(-200, y0 - 4, W + 200, y0), P("#8a3a1c"))
        facing = 1 if r % 2 == 0 else -1
        for i, n in enumerate(cast[r]):
            X = 140 + i * 265 if facing > 0 else W - 140 - i * 265
            delay = (i * 0.125) if r == 1 else 0.0
            dancer(c, x, n, style, x.b - delay, X, y0 + reg_h - 8, 250, facing, PAINT_STYLE, i=i + r * 3,
                   energy=1.2)
            if i < 6:
                gx = X + 130 * facing
                pr.glyph_column(c, gx - 14, y0 + 6, reg_h * 0.6, 28, "#5a2412", seed=i + r * 10, a=0.75,
                                t=x.b / 4, lit=0.8 * x.kick, lit_col="#1f3f95")


def s_hook(c, x, variant=1):
    zoom = 1.06 + 0.05 * pulse(x.u, 7) * (0.5 + x.kick)
    rot = math.sin(x.b * math.pi / 4) * 1.4
    sx, sy = shake(x, 6 * x.kick)
    cam(c, zoom=zoom, rot=rot, ox=sx, oy=sy)
    wall_scene(c, x)
    # light sweep once per bar
    f = fract(x.b / 4)
    sxp = lerp(-600, W + 600, f)
    c.drawRect(skia.Rect.MakeLTRB(-400, -400, W + 400, H + 400),
               P(shader=lin_grad((sxp - 260, 0), (sxp + 260, 200), [(0, "#fff0c0", 0), (0.5, "#fff0c0", 0.22), (1, "#fff0c0", 0)]),
                 blend=BM.kScreen))
    c.restore()
    # torch-light vignette pulsing warm
    fx.vignette(c, 0.45 - 0.2 * x.kick, "#2a0e02")
    return default_post(bloom=0.3, threshold=0.62, chroma=4 * x.snare, flash=cut_flash(x, 0.3, 0.9),
                        flash_col="#fff4c0", sat=1.15)


# ----------------------------------------------------------------------------
# BREAK 1: tunnel dive into the pyramid
# ----------------------------------------------------------------------------
def s_break1(c, x):
    lb = x.lb
    speed = 1.2 + 1.6 * ease_in(clamp(lb / 8))
    zc = lb * speed * 1.4 + ease_in(clamp(lb / 8), 3) * 6
    rot = math.sin(lb * 0.8) * 4 + ease_in(clamp(lb / 8)) * 25
    c.save()
    c.translate(W / 2, H / 2)
    c.rotate(rot)
    c.drawRect(skia.Rect.MakeLTRB(-W, -H, W, H), P("#0a0503"))
    fz = 900.0
    spacing = 3.0
    first = math.floor(zc / spacing) + 1
    frames = [first + k for k in range(14)]
    fglyph = font("Hiero.ttf", 10)
    gl = pr.glyph_list(80, 42)
    for k in reversed(frames):
        z = k * spacing - zc
        if z < 0.2:
            continue
        s = fz / z
        fog = clamp(1 - z / 36) ** 1.3
        X, Y = 1.25 * s, 0.85 * s
        T = 0.16 * s
        # corridor walls between this frame and the next farther one
        z2 = z + spacing
        s2 = fz / z2
        X2, Y2 = 1.25 * s2, 0.85 * s2
        fog2 = clamp(1 - z2 / 36) ** 1.3
        torch_lit = 0.75 + 0.25 * fbm(x.t * 6 + k, k)
        wallc = mix("#0a0503", "#9a5a2a", fog * torch_lit)
        for quad, shade in (([(-X, -Y), (-X2, -Y2), (-X2, Y2), (-X, Y)], 0.9), ([(X, -Y), (X2, -Y2), (X2, Y2), (X, Y)], 0.9),
                            ([(-X, Y), (-X2, Y2), (X2, Y2), (X, Y)], 0.55), ([(-X, -Y), (-X2, -Y2), (X2, -Y2), (X, -Y)], 0.4)):
            c.drawPath(poly(quad), P(scale_c(wallc, shade)))
        # painted bands running along the side walls (strong perspective cue)
        for (v0, v1, bc) in ((-0.62, -0.55, "#1f3f95"), (-0.5, -0.46, "#f2c14e"), (0.3, 0.36, "#c0392b"), (0.4, 0.44, "#2ec4b6")):
            bcol = mix("#0a0503", bc, fog * 0.85)
            for side in (-1, 1):
                c.drawPath(poly([(side * X, v0 * Y / 0.85 * 0.85), (side * X2, v0 * Y2), (side * X2, v1 * Y2), (side * X, v1 * Y)]), P(bcol))
        # floor tiles
        c.drawLine(-X, Y, -X2, Y2, P(mix("#0a0503", "#5a3014", fog), 1, stroke=max(1, s * 0.01)))
        c.drawLine(X, Y, X2, Y2, P(mix("#0a0503", "#5a3014", fog), 1, stroke=max(1, s * 0.01)))
        # torch glow pools on the walls
        for side in (-1, 1):
            gx = side * lerp(X, X2, 0.3)
            pr.glow(c, gx, -0.1 * lerp(Y, Y2, 0.3), 0.9 * lerp(s, s2, 0.3), "#ff9a3a", 0.35 * fog * torch_lit)
        # door frame
        beat_lit = pulse(fract(x.b) if (k % 4) == (x.bi % 4) else 1.0, 5)
        fr = skia.Path()
        fr.addRect(skia.Rect.MakeLTRB(-X - T, -Y - T, X + T, Y + T))
        fr.addRect(skia.Rect.MakeLTRB(-X, -Y, X, Y))
        fr.setFillType(skia.PathFillType.kEvenOdd)
        fc = mix("#0a0503", mix("#b8793a", "#ffd27a", beat_lit), fog)
        c.drawPath(fr, P(fc))
        # lintel glyphs + frame bands
        c.drawRect(skia.Rect.MakeLTRB(-X - T, -Y - T, X + T, -Y - T * 0.8), P(mix("#0a0503", "#1f3f95", fog)))
        c.drawRect(skia.Rect.MakeLTRB(-X - T, Y + T * 0.8, X + T, Y + T), P(mix("#0a0503", "#1f3f95", fog)))
        if s > 12:
            fs = font("Hiero.ttf", T * 0.55)
            n = int((2 * X) / (T * 0.62))
            gcol = mix("#0a0503", mix("#4a2410", "#1f3f95", beat_lit), fog)
            for j in range(n):
                c.drawString(gl[(j + k * 3) % len(gl)], -X + j * T * 0.62, -Y - T * 0.22, fs, P(gcol))
            m = int((2 * Y) / (T * 0.62))
            for j in range(m):
                c.drawString(gl[(j + k * 5) % len(gl)], -X - T * 0.8, -Y + (j + 1) * T * 0.62, fs, P(gcol))
                c.drawString(gl[(j + k * 7) % len(gl)], X + T * 0.2, -Y + (j + 1) * T * 0.62, fs, P(gcol))
        # torches mounted on the side walls just behind each door
        if 0.3 < z < 30:
            for side in (-1, 1):
                tz = z + spacing * 0.3
                ts = fz / tz
                tx = side * 1.2 * ts
                pr.torch(c, tx, 0.05 * ts, 0.13 * ts, x.t, seed=k * 2 + side, a=clamp(1 - tz / 30))
    # far light
    end = ease_in(clamp((lb - 5) / 3), 2)
    pr.glow(c, 0, 0, 200 + 900 * end, "#fff0c0", 0.3 + 0.7 * end)
    c.restore()
    return default_post(bloom=0.5, threshold=0.5, flash=max(cut_flash(x, 0.25, 0.8), ease_in(clamp((lb - 6.5) / 1.5), 3)),
                        flash_col="#fff4d8", vignette=0.7, chroma=3 + 5 * x.kick)


# ----------------------------------------------------------------------------
# VERSE 2: night on the Nile
# ----------------------------------------------------------------------------
def night_sky(c, x, horizon=560):
    pr.sky(c, [(0, "#03051a"), (0.55, "#101a4a"), (0.85, "#2e2a6e"), (1, "#5a3a7a")], 0, horizon)
    pr.stars(c, x.t, alpha=1.0, seed=9, n=900, h=horizon, tw=1.5)


def river(c, x, horizon, moon_x, a=1.0):
    c.drawRect(skia.Rect.MakeLTRB(-300, horizon, W + 300, H + 300),
               P(shader=lin_grad((0, horizon), (0, H), [(0, "#27306a", a), (0.4, "#0f2a4a", a), (1, "#04101e", a)])))
    rng = np.random.default_rng(1)
    for i in range(160):
        yy = horizon + 4 + (i / 160) ** 1.6 * (H - horizon)
        spread = 20 + (yy - horizon) * 0.35
        xx = moon_x + rng.normal(0, 1) * spread + math.sin(x.t * 1.5 + i) * 8
        w = 10 + (yy - horizon) * 0.12 * rng.uniform(0.5, 1.5)
        al = (0.35 + 0.65 * (0.5 + 0.5 * math.sin(x.t * 3 + i * 1.3))) * a
        c.drawLine(xx - w, yy, xx + w, yy, P("#fff4d0", al * 0.8, stroke=1.5 + (yy - horizon) * 0.006))
    for i in range(40):
        yy = horizon + 20 + i * 14
        xx = (i * 211 + x.t * 20) % (W + 200) - 100
        c.drawLine(xx, yy, xx + 60 + i * 2, yy, P("#6a7ac0", 0.15 * a, stroke=1.5))


def fireflies(c, x, n=30, region=(0, 300, W, 800), seed=0):
    for i in range(n):
        X = region[0] + (hash1(i + seed) * (region[2] - region[0]) + math.sin(x.t * 0.7 + i) * 60) % (region[2] - region[0])
        Y = region[1] + hash1(i * 3 + seed) * (region[3] - region[1]) + math.sin(x.t * 1.1 + i * 2) * 30
        a = 0.5 + 0.5 * math.sin(x.t * 4 + i * 1.7)
        pr.glow(c, X, Y, 16, "#d8ff7a", 0.7 * a)
        c.drawCircle(X, Y, 2.2, P("#f8ffd0", a))


def reeds(c, x, x0, y, n=14, h=260, color="#06140e", flip=1):
    for i in range(n):
        X = x0 + i * 16 * flip
        sway = math.sin(x.t * 1.2 + i * 0.5) * 12
        hh = h * (0.6 + 0.4 * hash1(i + 3))
        pth = skia.Path()
        pth.moveTo(X, y)
        pth.quadTo(X + sway * 0.3, y - hh * 0.5, X + sway, y - hh)
        c.drawPath(pth, P(color, 1, stroke=4))
        c.drawOval(skia.Rect.MakeLTRB(X + sway - 14, y - hh - 10, X + sway + 14, y - hh + 12), P(color))


def s_verse2(c, x):
    idx, sb = x.shot([0, 4, 8, 13])
    post = default_post(bloom=0.4, threshold=0.64, flash=cut_flash(x, 0.6, 0.8), flash_col="#ffffff",
                        sat=1.1, tint=(0.95, 1.0, 1.08))
    if idx == 0:
        horizon = 600
        cam(c, zoom=1.0 + 0.01 * sb / 16)
        night_sky(c, x, horizon)
        pr.moon(c, 1450, 250, 105)
        for (px, w, h) in ((260, 420, 200), (560, 260, 120), (1650, 340, 150)):
            pr.pyramid(c, px, horizon + 2, w, h, lit="#3a3470", shade="#1c1a40", light=1, courses=False, cap=0.2)
        for i in range(9):
            pr.palm(c, 700 + i * 95 + (i % 3) * 20, horizon + 4, 110 + (i % 4) * 25, x.t, lean=0.08 * ((i % 3) - 1),
                    sil=True, color="#0a0a22", seed=i)
        c.drawRect(skia.Rect.MakeLTRB(-100, horizon - 6, W + 100, horizon + 6), P("#0a0a22"))
        river(c, x, horizon, 1450)
        bx = lerp(420, 980, sb / 16)
        by = 790
        rock = math.sin(x.t * 1.3) * 2
        pr.felucca(c, bx, by, 95, x.t, rock=rock, sail="#b9b3d6")
        for i, n in enumerate(["nefi", "heru", "miu"]):
            dancer(c, x, n, "toggle", x.b - i * 0.25, bx - 110 + i * 85, by - 20, 150, i=i)
        for (lx, ly) in ((bx - 140, by - 110), (bx + 160, by - 120)):
            sw = math.sin(x.t * 2 + lx) * 6
            pr.glow(c, lx + sw, ly, 70, "#ffb03a", 0.6 + 0.3 * x.kick)
            c.drawCircle(lx + sw, ly, 7, P("#ffe6a0"))
        fireflies(c, x, 25, (0, 450, W, 700))
        for i in range(5):
            lx = 150 + i * 390
            op = 0.3 + 0.7 * hit(fract(x.b + i * 0.25), 0.4) if (x.bi + i) % 2 == 0 else 1 - 0.7 * hit(fract(x.b + i * 0.25), 0.4)
            pr.lotus(c, lx + math.sin(x.t + i) * 10, 1010 + (i % 2) * 30, 40, open_=op)
        reeds(c, x, -20, H + 20, 16, 340)
        reeds(c, x, W + 20, H + 20, 16, 300, flip=-1)
        c.restore()
    elif idx == 1:
        horizon = 700
        rock = math.sin(x.t * 1.3) * 2.5
        cam(c, zoom=1.0, rot=rock * 0.4, oy=math.sin(x.t * 1.3) * 10)
        night_sky(c, x, horizon)
        pr.moon(c, 380, 230, 120)
        river(c, x, horizon, 380)
        bx, by = W / 2 + 60, 900
        pr.felucca(c, bx, by, 290, x.t, rock=rock, sail="#a9a3c8")
        cast = ["nefi", "heru", "miu"]
        styles = ["offer", "toggle", "praise"]
        for i, n in enumerate(cast):
            dancer(c, x, n, styles[i], x.b - i * 0.5, bx - 330 + i * 260, by - 58, 360, i=i)
        for j, (lx, ly) in enumerate(((bx - 460, by - 330), (bx + 470, by - 360))):
            sw = math.sin(x.t * 2 + j) * 10
            c.drawLine(lx, ly - 80, lx + sw, ly, P("#3b2412", 1, stroke=3))
            pr.glow(c, lx + sw, ly + 18, 150, "#ffb03a", 0.55 + 0.35 * x.kick)
            c.drawRoundRect(skia.Rect.MakeLTRB(lx + sw - 16, ly, lx + sw + 16, ly + 40), 6, 6, P("#ffcf6a"))
        fireflies(c, x, 40, (0, 200, W, 900), seed=5)
        c.restore()
    else:
        wl = 330
        size = 440
        stride = 0.17
        cam_adv = stride * x.b
        scroll = cam_adv * size
        end = clamp((x.lbar - 12))
        cam(c, zoom=1.0, oy=-ease_in(end, 2) * 300)
        night_sky(c, x, wl)
        pr.moon(c, 1500, 130, 70, glow_a=0.4)
        pr.felucca(c, W / 2 - 200, wl, 120, x.t, rock=math.sin(x.t) * 2, sail="#b9b3d6")
        c.drawRect(skia.Rect.MakeLTRB(-300, wl, W + 300, H + 400),
                   P(shader=lin_grad((0, wl), (0, H), [(0, "#1a6a7a", 1), (0.5, "#0c3a52", 1), (1, "#051624", 1)])))
        # god rays
        for i in range(7):
            X = 200 + i * 260 + math.sin(x.t * 0.4 + i) * 40
            c.drawPath(poly([(X, wl), (X + 70, wl), (X + 260, H), (X + 90, H)]), P("#bff4ff", 0.06 + 0.03 * math.sin(x.t + i)))
        c.drawLine(-300, wl, W + 300, wl, P("#bff4ff", 0.6, stroke=3))
        # fish schools
        for sch in range(3):
            for f in range(9):
                fxp = (f * 38 + sch * 700 + x.t * (90 + sch * 30) + math.sin(x.t * 2 + f) * 20) % (W + 400) - 200
                fyp = 470 + sch * 110 + math.sin(x.t * 1.5 + f * 0.7 + sch) * 30 + (f % 3) * 18
                fish = skia.Path()
                fish.addOval(skia.Rect.MakeLTRB(fxp - 16, fyp - 6, fxp + 16, fyp + 6))
                c.drawPath(fish, P(["#ffb03a", "#2ec4b6", "#ff7eb6"][sch], 0.85))
                c.drawPath(poly([(fxp - 14, fyp), (fxp - 26, fyp - 8), (fxp - 26, fyp + 8)]), P(["#ffb03a", "#2ec4b6", "#ff7eb6"][sch], 0.85))
        # riverbed
        bed = 1000
        pr.dune(c, scroll, bed, 10, 7.0, "#3a4a3a", "#1a221a")
        for i in range(12):
            X = (i * 260 - scroll) % (W + 300) - 150
            reeds(c, x, X, bed + 10, 3, 200 + (i % 3) * 60, color="#0e3a2a")
        landing_puffs(c, x, W / 2 - 100, bed, size, x.b, scroll_adv=cam_adv, color="#9ab0a0", a=0.5)
        dancer(c, x, "sobi", "egypt_walk", x.b, W / 2 - 100, bed, size, scroll_adv=cam_adv, energy=1.3)
        # bubbles
        rng = np.random.default_rng(4)
        nb = 40 + int(160 * end)
        for i in range(nb):
            bx0 = rng.uniform(0, W)
            sp = rng.uniform(80, 220) * (1 + 3 * end)
            ph = rng.uniform(0, 10)
            yy = H + 50 - ((x.t + ph) * sp) % (H - wl + 100)
            r = rng.uniform(3, 12) * (1 + end)
            c.drawCircle(bx0 + math.sin(x.t * 3 + i) * 10, yy, r, P("#dff8ff", 0.5, stroke=1.5))
        c.restore()
        post.update(flash=max(cut_flash(x, 0.4, 0.5) if sb < 1 else 0.0, ease_in(end, 3) * 0.9),
                    flash_col="#dff8ff", tint=(0.9, 1.0, 1.1))
    return post


# ----------------------------------------------------------------------------
# HOOK 2: sphinx laser stage
# ----------------------------------------------------------------------------
def s_hook2(c, x):
    sx, sy = shake(x, 8 * x.kick)
    zoom = 1.0 + 0.04 * pulse(x.u, 8)
    cam(c, zoom=zoom, ox=sx, oy=sy)
    horizon = 700
    pr.sky(c, [(0, "#070018"), (0.5, "#2a0a4a"), (0.85, "#7a1a6a"), (1, "#ff3a7a")], 0, horizon)
    pr.stars(c, x.t, 0.8, seed=12, n=400, h=500)
    for (px, w, h, nc) in ((300, 560, 330, "#00e5ff"), (1560, 700, 420, "#ff2e88"), (1080, 380, 200, "#b8ff3c")):
        pr.pyramid(c, px, horizon, w, h, lit="#2a1440", shade="#12081e", light=-1, courses=False,
                   neon=nc, neon_a=0.6 + 0.4 * x.kick)
    eye = pr.sphinx(c, 620, horizon + 30, 190, eye_glow=0.6 + 0.4 * x.kick, lit="#c98a5a", shade="#6a3a3a")
    # lasers
    cols = ["#ff2e88", "#00e5ff", "#b8ff3c", "#ffd35a"]
    for j in range(4):
        ang = math.radians(-30 + 40 * math.sin(x.b * math.pi / 2 + j * 1.3) - j * 8)
        L = 2600
        ex, ey = eye[0] + math.cos(ang) * L, eye[1] + math.sin(ang) * L
        cc = cols[(j + x.bi) % 4]
        c.drawLine(eye[0], eye[1], ex, ey, P(cc, 0.35, stroke=18, blur=10))
        c.drawLine(eye[0], eye[1], ex, ey, P(mix(cc, "white", 0.6), 0.9, stroke=3))
    # stage
    c.drawRect(skia.Rect.MakeLTRB(-200, horizon + 30, W + 200, H + 200), P(shader=lin_grad((0, horizon + 30), (0, H), [(0, "#2a1030", 1), (1, "#0a0410", 1)])))
    stage_y = 930
    c.drawRect(skia.Rect.MakeLTRB(80, stage_y, W - 80, stage_y + 40), P("#f2c14e"))
    for i in range(20):
        on = (i + x.bi) % 4 == 0
        c.drawRect(skia.Rect.MakeLTRB(90 + i * 88, stage_y + 8, 150 + i * 88, stage_y + 32), P("#ff2e88" if on else "#5a1a4a"))
    # spotlights
    for j, (sx0, cc) in enumerate(((150, "#00e5ff"), (W - 150, "#ff2e88"), (W / 2, "#fff4c0"))):
        tx = W / 2 + math.sin(x.b * math.pi / 4 + j * 2) * 600
        c.drawPath(poly([(sx0 - 20, -50), (sx0 + 20, -50), (tx + 170, stage_y), (tx - 170, stage_y)]),
                   P(shader=lin_grad((0, 0), (0, stage_y), [(0, cc, 0.35), (1, cc, 0.05)]), blend=BM.kScreen))
        c.drawOval(skia.Rect.MakeLTRB(tx - 170, stage_y - 18, tx + 170, stage_y + 18), P(cc, 0.3, blend=BM.kScreen))
    cast = ["khet", "nefi", "anpu", "miu", "tut"]
    for i, n in enumerate(cast):
        dancer(c, x, n, "hook", x.b, 260 + i * 350, stage_y, 330, i=i, energy=1.3)
    # crowd
    for row in range(3):
        for i in range(28):
            X = i * 72 + row * 36 - 40
            bob = abs(math.sin((x.b + i * 0.13 + row * 0.3) * math.pi)) * 14 * (0.5 + x.kick)
            Y = H - 40 + row * 45 - bob
            c.drawCircle(X, Y - 30, 26, P("#07020c"))
            c.drawOval(skia.Rect.MakeLTRB(X - 42, Y - 5, X + 42, Y + 80), P("#07020c"))
            if (i + row + x.bi) % 7 == 0:
                c.drawLine(X + 20, Y - 10, X + 45, Y - 110 - bob, P("#07020c", 1, stroke=16))
    c.restore()
    return default_post(bloom=0.55, threshold=0.5, chroma=5 * x.snare, flash=cut_flash(x, 0.3, 0.9),
                        flash_col="#ff9ad0", sat=1.15)


# ----------------------------------------------------------------------------
# BREAK 2: scarab swarm forms the eye
# ----------------------------------------------------------------------------
def _bez_q(p0, p1, p2, n):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in np.linspace(0, 1, n)]


def _bez_c(p0, p1, p2, p3, n):
    out = []
    for t in np.linspace(0, 1, n):
        mt = 1 - t
        out.append((mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0],
                    mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1]))
    return out


def eye_targets():
    pts = []
    pts += _bez_q((-1.05, -0.55), (0.0, -0.95), (1.25, -0.62), 26)
    pts += _bez_q((-0.95, 0.02), (-0.1, -0.8), (0.8, -0.02), 22)
    pts += _bez_q((-0.95, 0.02), (-0.1, 0.45), (0.8, -0.02), 20)
    pts += [(lerp(0.8, 1.55, k), -0.02) for k in np.linspace(0, 1, 9)]
    pts += _bez_q((-0.15, 0.3), (-0.12, 0.7), (-0.3, 1.15), 11)
    pts += _bez_c((0.25, 0.28), (0.35, 0.8), (1.05, 1.15), (1.2, 0.75), 16)
    pts += _bez_c((1.2, 0.75), (1.3, 0.45), (0.95, 0.4), (0.92, 0.62), 8)
    for a in np.linspace(0, TAU, 14, endpoint=False):
        pts.append((-0.08 + math.cos(a) * 0.18, -0.05 + math.sin(a) * 0.18))
    return pts


def s_break2(c, x):
    lb = x.lb
    c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P(shader=rad_grad((W / 2, H / 2), W * 0.7, [(0, "#4a2a12", 1), (1, "#0e0703", 1)])))
    sand = cached_img("sand_dark", lambda: sandstone(W, H, (0.35, 0.22, 0.12), 5))
    p = skia.Paint()
    p.setAlphaf(0.5)
    p.setBlendMode(BM.kMultiply)
    c.drawImage(sand, 0, 0, fx.SAMP, p)
    targets = cached_img("eye_targets", eye_targets)
    S = 330
    cx, cy = W / 2 - 60, H / 2 - 20
    form = ease_io(clamp((lb + 0.4) / 4.2))
    burst = ease_out(clamp((lb - 6.5) / 1.3), 2)
    eye_solid = clamp((lb - 4.0) / 1.0) * (1 - burst)
    if eye_solid > 0:
        pr.glow(c, cx, cy, 700, "#ffb03a", 0.45 * eye_solid * (0.7 + 0.3 * x.kick))
        pr.eye_of_horus(c, cx, cy, S, color="#f2c14e", a=eye_solid, fill_iris=False, stroke_k=0.8,
                        glow_col="#ffd35a", open_=0.3 + 0.7 * ease_out(clamp((lb - 4) / 1.5)))
    rng = np.random.default_rng(8)
    n = len(targets)
    for i in range(n):
        tx, ty = targets[i]
        tx, ty = cx + tx * S, cy + ty * S
        a0 = rng.uniform(0, TAU)
        r0 = rng.uniform(0.55, 0.95) * W * 0.7
        sx0, sy0 = cx + math.cos(a0) * r0, cy + math.sin(a0) * r0
        delay = rng.uniform(0, 0.35)
        f = ease_io(clamp((form - delay) / (1 - delay)))
        # curved approach
        mx, my = lerp(sx0, tx, 0.5) + math.sin(a0 * 3) * 300, lerp(sy0, ty, 0.5) + math.cos(a0 * 2) * 300
        px = (1 - f) ** 2 * sx0 + 2 * (1 - f) * f * mx + f * f * tx
        py = (1 - f) ** 2 * sy0 + 2 * (1 - f) * f * my + f * f * ty
        # jitter while holding
        px += math.sin(x.t * 9 + i) * 3 * f
        py += math.cos(x.t * 7 + i) * 3 * f
        if burst > 0:
            ang = math.atan2(ty - cy, tx - cx) + rng.uniform(-0.4, 0.4)
            d = burst * rng.uniform(900, 1500)
            px += math.cos(ang) * d
            py += math.sin(ang) * d - burst * 200
        # heading
        f2 = clamp(f + 0.02)
        qx = (1 - f2) ** 2 * sx0 + 2 * (1 - f2) * f2 * mx + f2 * f2 * tx
        qy = (1 - f2) ** 2 * sy0 + 2 * (1 - f2) * f2 * my + f2 * f2 * ty
        ang = math.degrees(math.atan2(qy - py, qx - px)) + 90 if f < 0.98 else -20 + 40 * hash1(i)
        wings = 1.0 if (f < 0.95 or burst > 0) else 0.2 * (0.5 + 0.5 * math.sin(x.t * 30 + i))
        pr.scarab(c, px, py, 22 if burst == 0 else 22 + 20 * burst, ang, wings, gold=(i % 3 == 0) or eye_solid > 0.5,
                  t=x.t + i)
    return default_post(bloom=0.5, threshold=0.5, flash=max(cut_flash(x, 0.3, 0.6), 0.9 * pulse(lb - 6.5, 4) if lb > 6.5 else 0),
                        flash_col="#ffe9a0", chroma=3 * x.snare)


# ----------------------------------------------------------------------------
# NEON section
# ----------------------------------------------------------------------------
NEON_COLS = ["#00e5ff", "#ff2e88", "#b8ff3c", "#ffd35a", "#b06bff"]


def neon_style(i):
    return rig.Style("neon", glow=NEON_COLS[i % len(NEON_COLS)], ink_w=0.011)


def synth_bg(c, x, horizon=620, speed=1.0):
    pr.sky(c, [(0, "#05000f"), (0.45, "#2a0550"), (0.8, "#9a1a7a"), (1, "#ff4a8a")], 0, horizon)
    pr.stars(c, x.t, 0.8, seed=21, n=300, h=horizon - 150)
    pr.sun(c, W / 2, horizon - 150, 280, core="#fff27a", edge="#ff2e88", glow_c="#ff3a9a", glow_a=0.5 + 0.3 * x.kick,
           stripes=9, stripe_t=x.t * 0.5)
    for (px, w, h, nc) in ((360, 620, 340, "#00e5ff"), (1560, 620, 340, "#00e5ff"), (140, 300, 160, "#b06bff"), (1780, 300, 160, "#b06bff")):
        pr.pyramid(c, px, horizon, w, h, lit="#140428", shade="#0a0214", light=1, courses=False, neon=nc,
                   neon_a=0.7 + 0.3 * x.kick)
    pr.grid_floor(c, horizon, x.t, speed=speed, color="#ff2e88", a=1.0)


def s_neon(c, x):
    idx, sb = x.shot([0, 4, 6, 8])
    post = default_post(bloom=0.6, threshold=0.45, flash=cut_flash(x, 0.35, 0.9), flash_col="#ff9ad0", sat=1.2,
                        scan=0.1, chroma=2 + 4 * x.snare)
    if idx == 0:
        cam(c, zoom=1.0 + 0.03 * pulse(x.u, 7))
        synth_bg(c, x, speed=1.0 / A.period / 2)
        for side in (-1, 1):
            for j in range(2):
                X = W / 2 + side * (760 + j * 170)
                pr.palm(c, X, 640 + j * 60, 380 + j * 80, x.t, lean=-0.12 * side, sil=True, color="#12001f", seed=j)
                pr.palm(c, X, 640 + j * 60, 380 + j * 80, x.t, lean=-0.12 * side, sil=True, color="#ff2e88", seed=j, a=0.0)
        gy = 930
        names = ["miu", "anpu", "heru", "khet"]
        for i, n in enumerate(names):
            px = 420 + i * 360
            # reflection
            pose, adv = rig.dance("egypt_walk", x.b, i, stride=0.17)
            c.saveLayerAlpha(skia.Rect.MakeLTRB(px - 300, gy, px + 300, H), 60)
            c.save()
            c.translate(px, gy)
            c.scale(1, -0.6)
            rig.draw_dancer(c, n, pose, 0, 0, 320, 1, neon_style(i), t=x.t)
            c.restore()
            c.restore()
            rig.draw_dancer(c, n, pose, px, gy, 320, 1, neon_style(i), t=x.t)
        c.restore()
    elif idx in (1, 2):
        grid9 = idx == 2
        ncol = 3 if grid9 else 2
        nrow = 3 if grid9 else 2
        pw, ph = W / ncol, H / nrow
        names = ["miu", "anpu", "heru", "sobi", "nefi", "khet", "djehu", "tut", "meri"]
        styles = ["flex", "toggle", "praise", "hook", "offer", "up", "toggle", "hook", "flex"]
        bgs = [("#ff2e88", "#6a0a4a"), ("#00e5ff", "#063a5a"), ("#ffd35a", "#8a4a0a"), ("#b8ff3c", "#1a5a1a"),
               ("#b06bff", "#2a0a5a")]
        shift = x.bi // 4
        for r in range(nrow):
            for cidx in range(ncol):
                k = r * ncol + cidx
                ci = (k + shift) % len(names)
                x0, y0 = cidx * pw, r * ph
                c.save()
                c.clipRect(skia.Rect.MakeLTRB(x0 + 5, y0 + 5, x0 + pw - 5, y0 + ph - 5), skia.ClipOp.kIntersect, True)
                b1, b2 = bgs[(k + shift) % len(bgs)]
                ccx, ccy = x0 + pw / 2, y0 + ph / 2
                c.drawRect(skia.Rect.MakeLTRB(x0, y0, x0 + pw, y0 + ph), P(b2))
                # rotating sunburst
                c.save()
                c.translate(ccx, ccy + ph * 0.1)
                c.rotate(x.t * 20 * (1 if k % 2 else -1))
                for j in range(12):
                    c.save()
                    c.rotate(j * 30)
                    c.drawPath(poly([(0, 0), (1400, -180), (1400, 180)]), P(b1, 0.35))
                    c.restore()
                c.restore()
                pr.glyph_row(c, x0 + 10, y0 + 40, pw, 30, b1, seed=k, a=0.7)
                size = ph * 0.9
                delay = k * 0.125 if grid9 else 0.0
                dancer(c, x, names[ci], styles[ci], x.b - delay, ccx - size * 0.05, y0 + ph - 8, size, i=k, energy=1.3)
                c.restore()
                # flashing border
                on = (k + x.bi) % (ncol * nrow) == 0
                c.drawRect(skia.Rect.MakeLTRB(x0 + 5, y0 + 5, x0 + pw - 5, y0 + ph - 5),
                           P("#ffffff" if on else "#000000", 1, stroke=10 if on else 10))
        post.update(bloom=0.35, threshold=0.6, scan=0.06, flash=cut_flash(x, 0.25, 0.7) if sb < 1 else 0.3 * pulse(sb % 4 if sb >= 0 else 1, 8) if sb % 4 < 0.5 else 0.0)
    else:
        # mummy solo in the tomb
        size = 520
        stride = 0.17
        cam_adv = -stride * x.b
        cam(c, zoom=1.0 + 0.02 * pulse(x.u, 6))
        c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P(shader=lin_grad((0, 0), (0, H), [(0, "#07040e", 1), (0.8, "#1a0f1a", 1), (1, "#0a0508", 1)])))
        wall = cached_img("tombwall", lambda: sandstone(W + 800, H, (0.3, 0.2, 0.18), 17))
        scroll = cam_adv * size
        c.drawImage(wall, -((-scroll * 0.4) % 800), 0)
        c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P("#0a0412", 0.55))
        # sarcophagi pop open on bar downbeats
        for j in range(5):
            X = ((j * 420 + scroll * 0.4) % (5 * 420)) - 200
            opened = clamp((x.lbar - 8) - j * 0.5 + 0.2)
            sarcophagus(c, x, X, 820, 190, opened, ["nefi", "khet", "djehu", "meri", "tut"][j], j)
        c.drawRect(skia.Rect.MakeLTRB(0, 860, W, H), P("#140a10"))
        # spotlight
        c.drawPath(poly([(W / 2 - 40, -40), (W / 2 + 40, -40), (W / 2 + 360, 980), (W / 2 - 360, 980)]),
                   P(shader=lin_grad((0, 0), (0, 980), [(0, "#e8f0ff", 0.35), (1, "#e8f0ff", 0.08)]), blend=BM.kScreen))
        c.drawOval(skia.Rect.MakeLTRB(W / 2 - 380, 950, W / 2 + 380, 1010), P("#e8f0ff", 0.25, blend=BM.kScreen))
        # smoke
        for j in range(8):
            X = (j * 300 + x.t * 40) % (W + 600) - 300
            c.drawCircle(X, 960 + math.sin(x.t + j) * 20, 180, P("#b8a8d8", 0.08, blur=60))
        dancer(c, x, "mummy", "moonwalk", x.b, W / 2 - 60, 980, size, scroll_adv=cam_adv, energy=1.2)
        c.restore()
        post.update(scan=0.0, bloom=0.45, sat=0.9, tint=(0.95, 1.0, 1.1))
    return post


def sarcophagus(c, x, X, gy, w, opened, occupant, j):
    h = w * 2.6
    top = gy - h
    shell = skia.Path()
    shell.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(X - w / 2, top, X + w / 2, gy), w / 2, w / 2))
    c.drawPath(shell, P("#1a0f14"))
    if opened > 0:
        dancer(c, x, occupant, "toggle", x.b - j * 0.25, X, gy - 20, h * 0.8, i=j)
    # lid swings open (hinge on the left)
    c.save()
    c.translate(X - w / 2, 0)
    c.scale(1 - 0.92 * ease_out(opened), 1)
    c.translate(-(X - w / 2), 0)
    lid = skia.Path()
    lid.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(X - w / 2, top, X + w / 2, gy), w / 2, w / 2))
    c.drawPath(lid, P(shader=lin_grad((X - w / 2, 0), (X + w / 2, 0), [(0, "#b8801e", 1), (0.5, "#f2c14e", 1), (1, "#9a6a14", 1)])))
    c.save()
    c.clipPath(lid, skia.ClipOp.kIntersect, True)
    for k in range(10):
        yy = top + h * 0.3 + k * h * 0.07
        c.drawRect(skia.Rect.MakeLTRB(X - w / 2, yy, X + w / 2, yy + h * 0.03), P("#1f3f95" if k % 2 else "#2ec4b6"))
    c.drawOval(skia.Rect.MakeLTRB(X - w * 0.28, top + h * 0.06, X + w * 0.28, top + h * 0.24), P("#e8b86a"))
    c.drawCircle(X - w * 0.1, top + h * 0.14, w * 0.04, P("#120b08"))
    c.drawCircle(X + w * 0.1, top + h * 0.14, w * 0.04, P("#120b08"))
    c.restore()
    c.restore()


# ----------------------------------------------------------------------------
# BRIDGE: star map
# ----------------------------------------------------------------------------
def milky_way():
    s = skia.Surface(W + 600, H + 600)
    c = s.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    rng = np.random.default_rng(31)
    for i in range(90):
        f = rng.uniform(0, 1)
        X = f * (W + 600)
        Y = (1 - f) * (H + 600) * 0.9 + rng.normal(0, 90)
        r = rng.uniform(60, 220)
        cc = ["#6a4ac0", "#3a5ad0", "#c05aa0", "#4ac0c0"][i % 4]
        c.drawCircle(X, Y, r, P(cc, rng.uniform(0.04, 0.1), blur=r * 0.6))
    for i in range(2600):
        f = rng.uniform(0, 1)
        X = f * (W + 600)
        Y = (1 - f) * (H + 600) * 0.9 + rng.normal(0, 140)
        c.drawCircle(X, Y, rng.uniform(0.4, 1.4), P("#fff6e8", rng.uniform(0.2, 0.9)))
    return s.makeImageSnapshot()


CONSTELLATION = [  # (name, x, y) in unit space: an Egyptian-pose figure
    ("head", 0.05, -0.95), ("neck", 0.0, -0.72), ("shf", 0.18, -0.66), ("elf", 0.52, -0.66), ("hdf", 0.52, -1.05),
    ("tipf", 0.75, -1.05), ("shb", -0.18, -0.66), ("elb", -0.52, -0.66), ("hdb", -0.52, -0.3), ("tipb", -0.78, -0.3),
    ("pel", 0.0, -0.1), ("knf", 0.18, 0.3), ("ftf", 0.3, 0.72), ("knb", -0.15, 0.32), ("ftb", -0.25, 0.72)]
CONST_LINES = [("head", "neck"), ("neck", "shf"), ("shf", "elf"), ("elf", "hdf"), ("hdf", "tipf"), ("neck", "shb"),
               ("shb", "elb"), ("elb", "hdb"), ("hdb", "tipb"), ("neck", "pel"), ("pel", "knf"), ("knf", "ftf"),
               ("pel", "knb"), ("knb", "ftb")]


def s_bridge(c, x):
    lb = x.lb
    tilt = ease_io(clamp(lb / 16))
    c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P(shader=lin_grad((0, 0), (0, H), [(0, "#01020a", 1), (0.7, "#0a1030", 1), (1, "#1a1a48", 1)])))
    mw = cached_img("milky", milky_way)
    c.drawImage(mw, -300 - tilt * 120, -420 + tilt * 200)
    pr.stars(c, x.t, 1.0, seed=33, n=700, tw=0.8)
    # constellation dancer: arms toggle on the last bar
    pts = {n: (xx, yy) for n, xx, yy in CONSTELLATION}
    flip = clamp((lb - 12) * 2) if lb > 12 else 0
    if lb > 12:
        k = math.floor(lb - 12)
        ph = hit(fract(lb - 12), 0.3)
        a_ = 1 if k % 2 == 0 else -1
        m = lerp(-a_, a_, ph)
        pts["hdf"] = (0.52, -0.66 - 0.39 * m)
        pts["tipf"] = (0.75, -0.66 - 0.39 * m)
        pts["hdb"] = (-0.52, -0.66 + 0.36 * m)
        pts["tipb"] = (-0.78, -0.66 + 0.36 * m)
    S = 330
    ox, oy = W / 2 + 60, 430 + tilt * 60
    prog = clamp(lb / 10)
    nl = len(CONST_LINES)
    for i, (a_, b_) in enumerate(CONST_LINES):
        f = clamp(prog * nl - i)
        if f <= 0:
            continue
        p0 = (ox + pts[a_][0] * S, oy + pts[a_][1] * S)
        p1 = (ox + pts[b_][0] * S, oy + pts[b_][1] * S)
        pe = (lerp(p0[0], p1[0], f), lerp(p0[1], p1[1], f))
        c.drawLine(p0[0], p0[1], pe[0], pe[1], P("#9ad0ff", 0.35, stroke=6, blur=4))
        c.drawLine(p0[0], p0[1], pe[0], pe[1], P("#d8ecff", 0.8, stroke=1.6))
    for i, (n, _, _) in enumerate(CONSTELLATION):
        px, py = ox + pts[n][0] * S, oy + pts[n][1] * S
        tw = 0.6 + 0.4 * math.sin(x.t * 3 + i)
        on = clamp(prog * len(CONSTELLATION) * 1.1 - i)
        if on > 0:
            pr.glow(c, px, py, 26 + 30 * x.hat, "#bfe0ff", 0.8 * on * tw)
            c.drawCircle(px, py, 3.5 if n != "head" else 6, P("#ffffff", on))
    # shooting stars
    for j in range(3):
        st = 1.5 + j * 2.7
        age = x.lt - st
        if 0 < age < 0.9:
            sx0, sy0 = 300 + j * 500, 80 + j * 60
            L = age * 900
            c.drawLine(sx0 + L, sy0 + L * 0.35, sx0 + L - 220, sy0 + (L - 220) * 0.35, P("#ffffff", 0.8 * (1 - age / 0.9), stroke=2.5))
    # horizon: pyramid + lone watcher
    hy = 1000 - tilt * 60
    pr.pyramid(c, 1450, hy, 900, 420, lit="#0c0c24", shade="#07071a", light=-1, courses=False)
    pr.pyramid(c, 420, hy, 600, 250, lit="#0a0a20", shade="#060616", light=-1, courses=False)
    c.drawRect(skia.Rect.MakeLTRB(0, hy - 2, W, H + 200), P("#050512"))
    pose, _ = rig.dance("idle", x.b, 0)
    pose["htilt"] = -18
    rig.draw_dancer(c, "nefi", pose, 1450, hy - 418, 70, -1, SIL, "#050512", x.t)
    return default_post(bloom=0.6, threshold=0.45, flash=cut_flash(x, 0.5, 0.5), flash_col="#bfe0ff", grain=0.09,
                        tint=(0.92, 0.98, 1.1))


# ----------------------------------------------------------------------------
# DAWN: barque of Ra + parade
# ----------------------------------------------------------------------------
def dawn_sky(c, x, k, horizon):
    pr.sky(c, [(0, mix("#0a1030", "#3a4a9a", k)), (0.5, mix("#3a2a6a", "#f07a8a", k)), (0.85, mix("#8a4a7a", "#ffc08a", k)),
               (1, mix("#c07a7a", "#fff0c0", k))], 0, horizon)
    pr.stars(c, x.t, 1 - k, seed=41, n=500, h=horizon - 100)


def s_dawn(c, x):
    idx, sb = x.shot([0, 4, 8, 13])
    post = default_post(bloom=0.45, threshold=0.52, flash=cut_flash(x, 0.4, 0.7), flash_col="#fff0d0")
    if idx == 0:
        k = sb / 16 * 0.6
        horizon = 800
        cam(c, zoom=1.0)
        dawn_sky(c, x, k, horizon)
        for i in range(5):
            pr.cloud(c, (i * 480 + x.t * 25) % (W + 600) - 300, 250 + i * 60, 420, 70, "#ffb0c0", 0.25, seed=i + 3)
        bx = lerp(250, 1500, sb / 16)
        by = 420 + math.sin(x.t * 0.8) * 10
        pr.barque(c, bx, by, 105, x.t)
        dancer(c, x, "heru", "praise", x.b, bx + 150, by - 18, 120, i=0)
        dancer(c, x, "djehu", "toggle", x.b, bx - 170, by - 26, 110, i=1)
        for (ox, h) in ((300, 380), (560, 300), (1500, 420), (1760, 320)):
            pr.obelisk(c, ox, horizon, 44, h, lit="#1a1030", shade="#100820", glyph_col="#2a1a40")
        for (px, w, h) in ((950, 700, 300), (1300, 360, 150)):
            pr.pyramid(c, px, horizon, w, h, lit="#241638", shade="#140a24", light=1, courses=False)
        c.drawRect(skia.Rect.MakeLTRB(0, horizon, W, H), P(shader=lin_grad((0, horizon), (0, H), [(0, "#1c1030", 1), (1, "#0a0614", 1)])))
        c.restore()
    elif idx == 1:
        k = 0.6 + sb / 16 * 0.3
        cam(c, zoom=1.0, oy=math.sin(x.t * 0.8) * 12)
        dawn_sky(c, x, k, H)
        for i in range(8):
            sp = 160 + (i % 3) * 120
            X = W + 400 - ((x.t * sp + i * 377) % (W + 900))
            pr.cloud(c, X, 200 + (i * 131) % 700, 380 + (i % 3) * 120, 80, "#ffd0d8", 0.3, seed=i)
        bx, by = W / 2, 800
        pr.barque(c, bx, by, 330, x.t, sun_r=0.5)
        dancer(c, x, "heru", "up", x.b, bx + 420, by - 70, 380, i=0, energy=1.2)
        dancer(c, x, "djehu", "toggle", x.b - 0.5, bx - 460, by - 80, 360, i=1)
        for i in range(12):
            sx_ = bx + math.sin(x.t * 1.3 + i) * 700
            sy_ = 200 + (i * 97 % 500) + math.cos(x.t + i) * 30
            pr.glow(c, sx_, sy_, 18, "#fff0b0", 0.5 + 0.5 * math.sin(x.t * 4 + i))
        for i in range(4):
            X = W + 400 - ((x.t * 900 + i * 700) % (W + 1400))
            pr.cloud(c, X, 1040, 700, 110, "#ffe0e8", 0.55, seed=i + 20)
        c.restore()
    else:
        # parade along the avenue, zooming out to reveal everyone
        loc = x.lb - 32
        z = lerp(1.55, 0.92, ease_io(clamp(loc / 20)))
        size, stride, gy = 280, 0.17, 880
        cam_adv = stride * x.b
        scroll = cam_adv * size
        cam(c, zoom=z, cx=W / 2 + 200 * (1 - (z - 0.92) / 0.63), cy=720)
        horizon = 640
        pr.sky(c, [(0, "#2a3480"), (0.45, "#b0507a"), (0.8, "#f08a5a"), (1, "#ffc890")], -600, horizon, x0=-1200, x1=W + 1200)
        pr.sun(c, W / 2 + 200, horizon + 20, 170, glow_a=0.45)
        pr.aten_rays(c, W / 2 + 200, horizon + 20, 170, x.t, a=0.5 + 0.4 * x.kick)
        c.drawRect(skia.Rect.MakeLTRB(-1200, horizon, W + 1200, H + 600), P(shader=lin_grad((0, horizon), (0, H), [(0, "#d88a5a", 1), (1, "#9a5a34", 1)])))
        # avenue of obelisks & ram-less sphinx plinths
        for j in range(16):
            X = (j * 360 - scroll * 0.6) % (16 * 360) - 1400
            pr.obelisk(c, X, 700, 40, 300, lit="#c88a52", shade="#8a5230", t=x.t, glow_top=0.4 * x.kick)
            c.drawRect(skia.Rect.MakeLTRB(X + 100, 690, X + 280, 720), P("#a8743e"))
            pr.sphinx(c, X + 200, 690, 28)
        c.drawRect(skia.Rect.MakeLTRB(-1200, 720, W + 1200, 740), P("#8a5a2e"))
        c.drawRect(skia.Rect.MakeLTRB(-1200, gy - 10, W + 1200, gy + 30), P("#d9a765"))
        for i in range(40):
            X = (i * 120 - scroll) % (40 * 120) - 1400
            c.drawLine(X, gy - 8, X + 40, gy + 28, P("#b8804a", 0.6, stroke=3))
        names = ["mummy", "sobi", "meri", "djehu", "tut", "heru", "miu", "anpu", "khet", "nefi"]
        for i, n in enumerate(names):
            px = -700 + i * 290
            long_shadow(c, x, n, "egypt_walk", x.b, px, gy, size, k=-1.2, a=0.22, i=i, scroll_adv=cam_adv)
        for i, n in reversed(list(enumerate(names))):
            px = -700 + i * 290
            landing_puffs(c, x, px, gy, size, x.b, scroll_adv=cam_adv, color="#f0d0a0")
            dancer(c, x, n, "egypt_walk", x.b - (i % 2) * 0.0, px, gy, size, i=i, scroll_adv=cam_adv, energy=1.2)
        if loc > 16:
            pr.confetti(c, x.t, seed=3, n=80, a=clamp((loc - 16) / 2), t0=x.s0 + A.period * 48)
        c.restore()
        end = clamp((x.lbar - 12))
        post.update(bloom=0.28, threshold=0.62, flash=max(cut_flash(x, 0.3, 0.5) if sb < 1 else 0, ease_in(end, 3) * 0.8))
    return post


# ----------------------------------------------------------------------------
# HOOK 3: kaleidoscope
# ----------------------------------------------------------------------------
def s_hook3(c, x):
    n = 8
    rot = x.t * 12 + 22.5 * ease_back(fract(x.b / 2) * 4 if fract(x.b / 2) < 0.25 else 1.0) * math.floor(x.b / 2)
    c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P(shader=rad_grad((W / 2, H / 2), W * 0.7, [(0, "#1f3f95", 1), (0.6, "#0b1740", 1), (1, "#03050f", 1)])))
    # expanding rings
    for j in range(6):
        r = ((x.b * 0.5 + j / 6) % 1.0) * 1300
        c.drawCircle(W / 2, H / 2, r, P("#f2c14e", 0.5 * (1 - r / 1300), stroke=8))
    wedge = poly([(0, 0), (1600 * math.cos(math.radians(-22.5)), 1600 * math.sin(math.radians(-22.5))),
                  (1600 * math.cos(math.radians(22.5)), 1600 * math.sin(math.radians(22.5)))])
    cast = ["nefi", "khet", "anpu", "miu", "heru", "djehu", "sobi", "tut"]
    kk = 1 + 0.08 * pulse(x.u, 6) * (0.5 + x.kick)
    for i in range(n):
        c.save()
        c.translate(W / 2, H / 2)
        c.rotate(rot + i * 45)
        if i % 2:
            c.scale(1, -1)
        c.clipPath(wedge, skia.ClipOp.kIntersect, True)
        c.scale(kk, kk)
        # content along the +x axis
        pr.glyph_row(c, 120, 12, 900, 34, "#f2c14e", seed=i % 2, a=0.8)
        c.save()
        c.translate(600, 0)
        c.rotate(90)
        dancer(c, x, cast[(i // 2 + x.bi // 4) % len(cast)], "hook", x.b, 0, 115, 240, i=i, energy=1.3)
        c.restore()
        pr.ankh(c, 880, -40, 60, "#2ec4b6", 0.9)
        pr.scarab(c, 300, -60, 30, 90, wings=0.5 + 0.5 * math.sin(x.t * 6), gold=True, t=x.t)
        c.drawCircle(1100, 0, 40 + 30 * x.kick, P("#ff5a1f", 0.8))
        c.restore()
    pr.sun(c, W / 2, H / 2, 120 + 25 * x.kick, core="#fff4c0", edge="#ff8a1f", glow_a=0.8)
    pr.eye_of_horus(c, W / 2 - 10, H / 2 + 5, 60, color="#1b0f0a", a=0.9, fill_iris=False)
    return default_post(bloom=0.55, threshold=0.5, chroma=3 + 6 * x.snare, flash=cut_flash(x, 0.3, 1.0),
                        flash_col="#ffffff", sat=1.2)


# ----------------------------------------------------------------------------
# HALL: shadow play among the columns
# ----------------------------------------------------------------------------
def s_hall(c, x):
    lb = x.lb
    scroll = lb * 40
    wall = cached_img("hallwall", lambda: sandstone(W + 1200, H, (0.62, 0.42, 0.26), 23))
    c.drawImage(wall, -(scroll * 0.5 % 1200), 0)
    # painted register on the wall
    for r in range(2):
        y0 = 120 + r * 360
        c.drawRect(skia.Rect.MakeLTRB(0, y0, W, y0 + 10), P("#1f3f95", 0.6))
        pr.glyph_row(c, -(scroll * 0.5 % 60), y0 + 60, W + 100, 40, "#4a2410", seed=r + 60, a=0.5)
    # torch
    tx, ty = W / 2 + 80, 1000
    flick = 1 + 0.15 * fbm(x.t * 5, 3)
    # shadows projected large on the wall
    names = ["khet", "miu", "anpu"]
    styles = ["toggle", "up", "hook"]
    c.saveLayerAlpha(skia.Rect.MakeLTRB(0, 0, W, H), 150)
    for i, n in enumerate(names):
        px = 380 + i * 520
        pose, _ = rig.dance(styles[i], x.b - i * 0.25, i)
        sx_ = W / 2 + (px - tx) * 1.9 * flick
        rig.draw_dancer(c, n, pose, sx_, 900, 820 * flick, 1, SIL, "#120804", x.t)
    c.restore()
    # warm light falloff
    c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P(shader=rad_grad((tx, ty - 200), 1400 * flick, [(0, "#ffb060", 0.25), (0.6, "#ff8030", 0.08), (1, "#000000", 0.55)])))
    c.drawRect(skia.Rect.MakeLTRB(0, 1000, W, H), P("#1a0c06"))
    # light shafts + dust
    for j in range(4):
        X = 300 + j * 450 - (scroll * 0.8) % 450
        c.drawPath(poly([(X, -20), (X + 90, -20), (X + 420, H), (X + 260, H)]), P("#ffe0a0", 0.07, blend=BM.kScreen))
    for j in range(60):
        dx_ = (hash1(j) * W + x.t * 20 * (0.5 + hash1(j + 9))) % W
        dy_ = (hash1(j + 3) * H + math.sin(x.t + j) * 20) % H
        c.drawCircle(dx_, dy_, 1.8, P("#fff0c0", 0.5 * (0.5 + 0.5 * math.sin(x.t * 3 + j))))
    # dancers in front of the torch
    for i, n in enumerate(names):
        px = 380 + i * 520
        dancer(c, x, n, styles[i], x.b - i * 0.25, px, 1040, 330, i=i, draw_style=rig.Style("flat", tint="#ff9a4a", tint_k=0.15))
    pr.torch(c, tx, ty + 40, 60, x.t, seed=4)
    # foreground columns sliding past
    for j in range(4):
        X = (j * 700 - scroll * 3.0) % (4 * 700) - 350
        pr.column(c, X, H + 40, 260, 1400, x.t, lit="#6a4020", shade="#2a1408", glyph_seed=j)
    # darken toward the stop
    end = ease_in(clamp((lb - 10) / 2), 2)
    return default_post(bloom=0.45, threshold=0.5, flash=cut_flash(x, 0.3, 0.7), flash_col="#ffd0a0", fade=end * 0.9,
                        sat=1.1, tint=(1.08, 1.0, 0.9))


# ----------------------------------------------------------------------------
# STOP: the freeze before the finale
# ----------------------------------------------------------------------------
def s_stop(c, x):
    lb = x.lb
    z = 1.0 + 0.05 * lb / 4 + ease_in(clamp((lb - 3.3) / 0.7), 4) * 3.5
    cam(c, zoom=z, cy=H * 0.55)
    c.drawRect(skia.Rect.MakeLTRB(-W, -H, W * 2, H * 2), P("#040306"))
    eo = ease_io(clamp(lb / 3.2))
    pr.eye_of_horus(c, W / 2 - 30, H / 2 - 120, 520, color="#2a1c10", a=0.6, fill_iris=False, open_=eo,
                    pupil_col="#f2c14e" if lb > 3.2 else "#2a1c10", glow_col="#ffb03a" if lb > 3 else None)
    c.drawPath(poly([(W / 2 - 60, -100), (W / 2 + 60, -100), (W / 2 + 330, 1000), (W / 2 - 330, 1000)]),
               P(shader=lin_grad((0, 0), (0, 1000), [(0, "#ffffff", 0.3), (1, "#ffffff", 0.05)]), blend=BM.kScreen))
    c.drawOval(skia.Rect.MakeLTRB(W / 2 - 330, 975, W / 2 + 330, 1025), P("#ffffff", 0.25))
    for j in range(40):
        dx_ = W / 2 - 250 + hash1(j) * 500
        dy_ = (hash1(j + 5) * 900 + x.t * 15) % 950
        c.drawCircle(dx_, dy_, 1.5, P("#ffffff", 0.5))
    pose, _ = rig.dance("freeze", 0.0, 0)
    pose["hx"] = 0.02 * math.sin(x.t * 2)
    pose["blink"] = 1.0 if 0.45 < fract(x.lt / 1.6) < 0.5 else 0.0
    rig.draw_dancer(c, "khet", pose, W / 2 - 30, 1000, 560, 1, t=x.t)
    c.restore()
    pupil_flash = pulse(lb - 3.4, 6) if lb > 3.4 else 0
    return default_post(bloom=0.5, threshold=0.5, sat=0.25 + 0.75 * ease_in(clamp((lb - 3.0))), vignette=0.8,
                        flash=max(0.0, ease_in(clamp((lb - 3.6) / 0.4), 2)), flash_col="#fff2c0",
                        fade=0.0 if lb > 0.3 else 0.0, chroma=6 * pupil_flash)


# ----------------------------------------------------------------------------
# FINALE
# ----------------------------------------------------------------------------
ALL = ["nefi", "khet", "anpu", "miu", "heru", "djehu", "sobi", "tut", "meri", "mummy"]


def fireworks_layer(c, x, strength=1.0):
    cols = ["#ffd35a", "#2ec4b6", "#ff5a8a", "#ffffff", "#8ad0ff"]
    shapes = ["ankh", "burst", "eye", "burst"]
    for j in range(-6, 1):
        bb = (x.bi // 2 + j) * 2
        tb = A.beat_time(bb)
        age = x.t - tb
        if age < 0:
            continue
        seed = bb * 13 + 5
        fxp = 300 + hash1(seed) * (W - 600)
        fyp = 150 + hash1(seed + 1) * 300
        launch = 0.35
        if age < launch:
            pr.rocket(c, fxp + 80, H, fxp, fyp, age / launch, a=strength)
        else:
            pr.firework(c, fxp, fyp, age - launch, seed=seed, color=cols[bb // 2 % len(cols)],
                        shape=shapes[bb // 2 % len(shapes)], a=strength, size=1.2, n=64)


def s_finale(c, x):
    idx, sb = x.shot([0, 3, 6])
    post = default_post(bloom=0.5, threshold=0.5, flash=cut_flash(x, 0.5, 1.0), flash_col="#fff6d0", sat=1.18,
                        chroma=3 * x.snare)
    if idx == 0:
        rise = ease_io(clamp(sb / 12))
        sxs, sys_ = shake(x, 5 * x.kick)
        cam(c, zoom=1.12 - 0.12 * rise, cy=H / 2 + 120 - 120 * rise, ox=sxs, oy=sys_)
        horizon = 640
        pr.sky(c, [(0, "#1a1050"), (0.4, "#a0306a"), (0.75, "#ff8a3a"), (1, "#ffe0a0")], -200, horizon)
        pr.sun(c, W / 2, horizon - 10, 230, glow_a=0.7 + 0.3 * x.kick, rays=1.0, ray_t=x.t)
        # searchlights from pyramid tips
        tips = []
        for (px, w, h) in ((380, 700, 330), (W - 380, 700, 330), (W / 2, 900, 430)):
            tips.append((px, horizon - h))
        for j, (tx_, ty_) in enumerate(tips):
            ang = math.radians(-90 + 35 * math.sin(x.b * math.pi / 4 + j * 2.1))
            L = 2000
            ex, ey = tx_ + math.cos(ang) * L, ty_ + math.sin(ang) * L
            nx, ny = -math.sin(ang) * 150, math.cos(ang) * 150
            c.drawPath(poly([(tx_, ty_), (ex + nx, ey + ny), (ex - nx, ey - ny)]),
                       P(shader=lin_grad((tx_, ty_), (ex, ey), [(0, "#fff0b0", 0.45), (1, "#fff0b0", 0)]), blend=BM.kScreen))
        fireworks_layer(c, x)
        for (px, w, h) in ((380, 700, 330), (W - 380, 700, 330), (W / 2, 900, 430)):
            pr.pyramid(c, px, horizon + 4, w, h, lit="#f6b565", shade="#8a4a2a", light=-1 if px > W / 2 else 1, cap=1.0 + x.kick)
        c.drawRect(skia.Rect.MakeLTRB(-300, horizon, W + 300, H + 400), P(shader=lin_grad((0, horizon), (0, H + 200), [(0, "#e39a5a", 1), (1, "#8a4a2a", 1)])))
        rows = [(740, 120, 16, 0.3), (860, 200, 10, 0.15), (1040, 330, 6, 0.0)]
        for r, (gy, size, cnt, dl) in enumerate(rows):
            sp = (W + 200) / cnt
            for i in range(cnt):
                n = ALL[(i + r * 3) % len(ALL)]
                px = -60 + i * sp + (r % 2) * sp * 0.5
                style = "hook" if r == 2 else "toggle"
                dancer(c, x, n, style, x.b - dl * (i % 3), px, gy, size, i=i + r * 5, energy=1.3)
        pr.confetti(c, x.t, seed=7, n=140, t0=x.s0)
        c.restore()
    elif idx == 1:
        # rapid close-up montage, a new character every two beats
        k = int(sb // 2)
        n = ALL[k % len(ALL)]
        bgc = [("#ff2e88", "#6a0a4a"), ("#00e5ff", "#063a5a"), ("#ffd35a", "#8a4a0a"), ("#2ec4b6", "#0b1740"),
               ("#ff7a3a", "#5a1a0a")][k % 5]
        zz = 1.0 + 0.05 * pulse(x.u, 6)
        cam(c, zoom=zz, rot=(-4 if k % 2 else 4) * (1 - hit(fract(sb / 2), 0.3)))
        c.drawRect(skia.Rect.MakeLTRB(-400, -400, W + 400, H + 400), P(bgc[1]))
        c.save()
        c.translate(W / 2, H * 0.62)
        c.rotate(x.t * 30)
        for j in range(16):
            c.save()
            c.rotate(j * 22.5)
            c.drawPath(poly([(0, 0), (2000, -200), (2000, 200)]), P(bgc[0], 0.35))
            c.restore()
        c.restore()
        pr.glyph_row(c, 0, 90, W + 100, 60, bgc[0], seed=k, a=0.9)
        pr.glyph_row(c, 0, H - 30, W + 100, 60, bgc[0], seed=k + 50, a=0.9)
        side = 1 if k % 2 == 0 else -1
        dancer(c, x, n, "hook", x.b, W / 2 - side * 60, H + 520, 1500, facing=side, i=k, energy=1.4)
        fx.speed_lines(c, x.t, a=0.35, n=30)
        c.restore()
        post.update(flash=0.55 * pulse(fract(sb / 2) * 2, 9) if fract(sb / 2) < 0.5 else 0.0, bloom=0.35,
                    chroma=4 + 4 * x.snare)
    else:
        # everyone together, big jump & mega firework
        cam(c, zoom=1.0 + 0.04 * pulse(x.u, 7))
        horizon = 700
        pr.sky(c, [(0, "#0b0b40"), (0.5, "#8a2a7a"), (0.85, "#ff7a3a"), (1, "#ffd08a")], 0, horizon)
        pr.sun(c, W / 2, horizon, 300, glow_a=0.8, rays=1.0, ray_t=x.t)
        pr.aten_rays(c, W / 2, horizon, 300, x.t, a=0.8)
        fireworks_layer(c, x)
        big = A.bar_time(80)
        if x.t > big:
            pr.firework(c, W / 2, 330, x.t - big, seed=99, color="#ffd35a", n=120, speed=900, size=1.6, shape="ankh")
            pr.firework(c, W / 2 - 500, 280, x.t - big - 0.15, seed=98, color="#2ec4b6", n=80, speed=700, size=1.2)
            pr.firework(c, W / 2 + 500, 280, x.t - big - 0.3, seed=97, color="#ff5a8a", n=80, speed=700, size=1.2)
        c.drawRect(skia.Rect.MakeLTRB(-300, horizon, W + 300, H + 300), P(shader=lin_grad((0, horizon), (0, H), [(0, "#e39a5a", 1), (1, "#8a4a2a", 1)])))
        for i, n in enumerate(ALL):
            px = 110 + i * 190
            style = "hook" if x.lbar < 6.75 else "up"
            dancer(c, x, n, style, x.b - (i % 2) * 0.5 * (1 if x.lbar < 6 else 0), px, 1010, 380, facing=1 if i < 5 else -1,
                   i=i, energy=1.4)
        pr.confetti(c, x.t, seed=11, n=200, t0=x.s0)
        c.restore()
        post.update(flash=max(cut_flash(x, 0.3, 0.8) if sb < 1 else 0.0, 0.8 * pulse(x.t - big, 3) if x.t > big else 0.0))
    return post


# ----------------------------------------------------------------------------
# OUTRO: into the sunset
# ----------------------------------------------------------------------------
def s_outro(c, x):
    lt = x.lt
    T = A.duration - x.s0
    p = clamp(lt / 12.0)
    horizon = 700
    pr.sky(c, [(0, mix("#1a1050", "#02030c", p)), (0.45, mix("#b0406a", "#140a30", p)), (0.8, mix("#ff8a3a", "#4a1a3a", p)),
               (1, mix("#ffe0a0", "#8a3a2a", p))], 0, horizon)
    pr.stars(c, x.t, alpha=clamp((lt - 4) / 5), seed=51, n=800, h=horizon)
    sun_y = lerp(horizon - 120, horizon + 160, ease_in(clamp(lt / 11), 1.5))
    c.save()
    c.clipRect(skia.Rect.MakeLTRB(0, 0, W, horizon), skia.ClipOp.kIntersect, True)
    pr.sun(c, W / 2 + 300, sun_y, 210, glow_a=0.6 * (1 - p * 0.7))
    c.restore()
    for (px, w, h) in ((W / 2 - 200, 820, 380), (W / 2 + 600, 520, 240), (W / 2 - 800, 400, 170)):
        pr.pyramid(c, px, horizon + 4, w, h, lit=mix("#6a3040", "#140a20", p), shade=mix("#2a1020", "#0a0512", p),
                   light=1, courses=False)
    c.drawRect(skia.Rect.MakeLTRB(0, horizon, W, H), P(shader=lin_grad((0, horizon), (0, H), [(0, mix("#c0704a", "#1a0e14", p), 1), (1, mix("#5a2a1a", "#07040a", p), 1)])))
    # walkers heading off toward the horizon
    names = ["nefi", "khet", "anpu", "miu", "heru", "djehu", "sobi", "mummy"]
    for i, n in reversed(list(enumerate(names))):
        prog = clamp((lt - i * 0.5) / 13)
        d = ease_in(prog, 1.3)
        size = lerp(300, 30, d)
        gy = lerp(1020, horizon + 6, d)
        px = lerp(200 + i * 160, W / 2 + 300 + (i - 3.5) * 14, d)
        if size < 34 and prog >= 1:
            continue
        a = 1 - clamp((prog - 0.85) / 0.15)
        long_shadow(c, x, n, "egypt_walk", x.b, px, gy, size, k=0.9, a=0.25 * a, i=i, scroll_adv=None)
        pose, _ = rig.dance("egypt_walk", x.b, i)
        rig.draw_dancer(c, n, pose, px, gy, size, 1, SIL, mix("#2a1418", "#07040a", p), x.t)
    # end title
    ta = clamp((x.t - 194.5) / 1.5) * (1 - clamp((x.t - 202.2) / 1.8))
    if ta > 0:
        c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P("#000000", 0.35 * ta))
        title(c, W / 2, 400, [1, 1, 1, 1], s=0.85, a=ta, shine=clamp((x.t - 196) / 2.5), glow_a=0.5)
    fade = clamp((x.t - 202.6) / 1.6)
    return default_post(bloom=0.45, threshold=0.5, fade=fade, vignette=0.6, flash=cut_flash(x, 0.4, 0.4))


SCENE_FN = {
    "intro": s_intro, "verse1": s_verse1, "hook1": s_hook, "break1": s_break1, "verse2": s_verse2,
    "hook2": s_hook2, "break2": s_break2, "neon": s_neon, "bridge": s_bridge, "dawn": s_dawn,
    "hook3": s_hook3, "hall": s_hall, "stop": s_stop, "finale": s_finale, "outro": s_outro,
}
