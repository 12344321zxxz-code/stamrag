"""Environments for CHASE (all original): practice room, night city, rooftops, train, tunnel, radio tower,
inside-the-blob dimension, dawn sky."""
import math

import numpy as np
import skia

from mv import props as mvp
from .core import (BM, H, TAU, W, C, P, clamp, col, ease_out, fbm, font, hash1, lerp, lin_grad, mix, path_from, poly,
                   rad_grad, rrect, scale_c, vnoise)
from . import cast as K
from . import fxk as FX

SAMP = skia.SamplingOptions(skia.FilterMode.kLinear)
_img = {}


def cached(key, fn):
    if key not in _img:
        _img[key] = fn()
    return _img[key]


# ----------------------------------------------------------------------------
# sky pieces
# ----------------------------------------------------------------------------
def sky(c, stops, y0=0, y1=H):
    mvp.sky(c, stops, y0, y1)


def stars(c, t, a=1.0, seed=1, n=600, h=H, tw=1.0):
    mvp.stars(c, t, alpha=a, seed=seed, n=n, h=h, tw=tw)


def moon(c, x, y, r, a=1.0, glow_a=0.5):
    mvp.moon(c, x, y, r, a, glow_a)


def glow(c, x, y, r, color, a=1.0):
    mvp.glow(c, x, y, r, color, a)


def cloud(c, x, y, w, h, color="#ffffff", a=0.5, seed=0):
    mvp.cloud(c, x, y, w, h, color, a, seed)


# ----------------------------------------------------------------------------
# city layers (pre-rendered strips)
# ----------------------------------------------------------------------------
CITY_STYLES = {
    # name: (seed, strip height, min h, max h, min w, max w, body, window colours, lit prob, win size, tint for night)
    "far": (11, 700, 180, 460, 70, 170, "#1b1747", ("#6a64c8", "#8a7ae8"), 0.22, 5),
    "mid": (23, 850, 250, 640, 120, 260, "#141036", ("#ffd27a", "#ffb35a", "#8af0ff"), 0.3, 8),
    "near": (37, 900, 300, 760, 180, 340, "#0d0a24", ("#ffd27a", "#ff9ad0", "#8af0ff"), 0.35, 11),
}


def make_city(style, width=4200):
    seed, sh, hmin, hmax, wmin, wmax, body, wins, prob, ws = CITY_STYLES[style]
    rng = np.random.default_rng(seed)
    s = skia.Surface(width, sh)
    c = s.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    x = -20.0
    beacons = []
    while x < width:
        bw = rng.uniform(wmin, wmax)
        bh = rng.uniform(hmin, hmax)
        top = sh - bh
        bcol = mix(body, "#000000", rng.uniform(0, 0.25))
        c.drawRect(skia.Rect.MakeLTRB(x, top, x + bw, sh), P(bcol))
        kind = rng.integers(0, 5)
        if kind == 0:  # stepped top
            c.drawRect(skia.Rect.MakeLTRB(x + bw * 0.2, top - bh * 0.08, x + bw * 0.8, top + 1), P(bcol))
        elif kind == 1:  # antenna
            ax = x + bw * rng.uniform(0.3, 0.7)
            c.drawLine(ax, top, ax, top - bh * 0.25, P(bcol, 1, stroke=3))
            beacons.append((ax, top - bh * 0.25))
        elif kind == 2:  # water tower
            tx = x + bw * rng.uniform(0.2, 0.6)
            c.drawRect(skia.Rect.MakeLTRB(tx, top - 40, tx + 36, top - 8), P(bcol))
            c.drawLine(tx + 4, top - 8, tx + 2, top, P(bcol, 1, stroke=3))
            c.drawLine(tx + 32, top - 8, tx + 34, top, P(bcol, 1, stroke=3))
            c.drawPath(poly([(tx - 3, top - 40), (tx + 18, top - 55), (tx + 39, top - 40)]), P(bcol))
        elif kind == 3:  # slanted roof
            c.drawPath(poly([(x, top), (x + bw, top), (x + bw, top - bh * 0.12)]), P(bcol))
        # windows
        cols_n = max(1, int((bw - 16) / (ws * 2.6)))
        rows_n = max(1, int((bh - 20) / (ws * 2.8)))
        wc = wins[rng.integers(0, len(wins))]
        for i in range(cols_n):
            for j in range(rows_n):
                if rng.random() < prob:
                    wx = x + 10 + i * ws * 2.6
                    wy = top + 14 + j * ws * 2.8
                    c.drawRect(skia.Rect.MakeLTRB(wx, wy, wx + ws * 1.4, wy + ws * 1.6), P(wc, rng.uniform(0.5, 1.0)))
        x += bw + rng.uniform(-10, 30)
    return s.makeImageSnapshot(), beacons


def city(c, style, scroll, y_bottom, t=0.0, a=1.0, scale=1.0, beacon=True):
    img, beacons = cached(("city", style), lambda: make_city(style))
    w, h = img.width(), img.height()
    off = scroll % w
    p = skia.Paint()
    p.setAlphaf(a)
    c.save()
    c.translate(0, y_bottom - h * scale)
    c.scale(scale, scale)
    for k in (-1, 0, 1, 2):
        x = k * w - off / scale
        if x > W / scale or x + w < 0:
            continue
        c.drawImage(img, x, 0, SAMP, p)
        if beacon:
            for bx, by in beacons:
                X = x + bx
                if -20 < X * scale < W + 20:
                    on = 0.5 + 0.5 * math.sin(t * 3 + bx)
                    c.drawCircle(X, by, 3.5, P("#ff3355", a * on))
                    c.drawCircle(X, by, 12, P("#ff3355", 0.25 * a * on, blur=6))
    c.restore()


# ----------------------------------------------------------------------------
# rooftop running track (near layer, drawn live)
# ----------------------------------------------------------------------------
class Rooftops:
    """A run of buildings with gaps placed where the hero needs to leap."""

    def __init__(self, leap_x, seed=5, base=820, start=-2000, end=30000, gap=190):
        rng = np.random.default_rng(seed)
        self.blocks = []
        x = start
        leaps = sorted(leap_x)
        li = 0
        while x < end:
            nxt = leaps[li] if li < len(leaps) else end + 1000
            bw = min(rng.uniform(600, 1100), nxt - x - gap / 2) if nxt - x > 300 else rng.uniform(600, 1100)
            if nxt - x <= 300:
                bw = max(80, nxt - x - gap / 2)
            top = base + rng.uniform(-70, 60)
            self.blocks.append((x, x + bw, top, int(rng.integers(0, 4))))
            x += bw
            if li < len(leaps) and abs(x - (nxt - gap / 2)) < 1:
                x = nxt + gap / 2
                li += 1
            elif li < len(leaps) and x >= nxt - gap / 2:
                x = nxt + gap / 2
                li += 1
            else:
                x += rng.uniform(0, 30)

    def roof_at(self, xw):
        for x0, x1, top, _ in self.blocks:
            if x0 <= xw <= x1:
                return top
        return None

    def draw(self, c, scroll, t, pal=("#1a1433", "#2a2150", "#8a7ae8")):
        body, edge, rim = pal
        for x0, x1, top, kind in self.blocks:
            X0, X1 = x0 - scroll, x1 - scroll
            if X1 < -50 or X0 > W + 50:
                continue
            c.drawRect(skia.Rect.MakeLTRB(X0, top, X1, H + 400), P(shader=lin_grad((0, top), (0, H), [(0, body, 1), (1, "#07040f", 1)])))
            c.drawRect(skia.Rect.MakeLTRB(X0 - 6, top - 10, X1 + 6, top + 8), P(edge))
            c.drawLine(X0 - 6, top - 10, X1 + 6, top - 10, P(rim, 0.8, stroke=2))
            # railing
            for k in range(int((X1 - X0) / 40)):
                xx = X0 + 20 + k * 40
                c.drawLine(xx, top - 10, xx, top - 38, P(edge, 1, stroke=3))
            c.drawLine(X0, top - 38, X1, top - 38, P(edge, 1, stroke=3))
            # rooftop props
            rng = np.random.default_rng(int(x0) & 0xffff)
            for k in range(2):
                px = X0 + rng.uniform(0.15, 0.85) * (X1 - X0)
                if kind == 0:
                    c.drawRect(skia.Rect.MakeLTRB(px, top - 70, px + 90, top - 10), P("#221a40"))
                    c.drawCircle(px + 45, top - 40, 22, P("#141030"))
                    for j in range(4):
                        a = t * 12 + j * 1.57
                        c.drawLine(px + 45, top - 40, px + 45 + math.cos(a) * 18, top - 40 + math.sin(a) * 18,
                                   P("#3a3070", 1, stroke=4))
                elif kind == 1:
                    c.drawLine(px, top - 10, px, top - 160, P("#221a40", 1, stroke=5))
                    c.drawLine(px - 30, top - 120, px + 30, top - 120, P("#221a40", 1, stroke=3))
                    on = 0.5 + 0.5 * math.sin(t * 4 + px * 0.01)
                    c.drawCircle(px, top - 162, 5, P("#ff3355", on))
                    c.drawCircle(px, top - 162, 16, P("#ff3355", 0.3 * on, blur=6))
                elif kind == 2:
                    c.drawRect(skia.Rect.MakeLTRB(px, top - 110, px + 70, top - 40), P("#2a2150"))
                    c.drawPath(poly([(px - 6, top - 110), (px + 35, top - 135), (px + 76, top - 110)]), P("#2a2150"))
                    c.drawLine(px + 8, top - 40, px + 4, top - 10, P("#2a2150", 1, stroke=4))
                    c.drawLine(px + 62, top - 40, px + 66, top - 10, P("#2a2150", 1, stroke=4))
            # windows on the facade below the roof line
            for j in range(4):
                for k in range(int((X1 - X0) / 70)):
                    if hash1(int(x0) + j * 31 + k * 7) < 0.45:
                        wx = X0 + 25 + k * 70
                        wy = top + 50 + j * 90
                        c.drawRect(skia.Rect.MakeLTRB(wx, wy, wx + 34, wy + 44), P("#ffd27a", 0.75))


# ----------------------------------------------------------------------------
# neon sign / billboard with lyric text
# ----------------------------------------------------------------------------
def neon_sign(c, x, y, w, h, words, t, color="#ff4fb0", a=1.0, fnt="Bangers.ttf", frame_col="#2a1a48", flick=0.0):
    c.drawRect(skia.Rect.MakeLTRB(x, y, x + w, y + h), P("#0a0716", 0.92 * a))
    c.drawRect(skia.Rect.MakeLTRB(x, y, x + w, y + h), P(frame_col, a, stroke=8))
    c.drawLine(x + w * 0.2, y + h, x + w * 0.2, y + h + 60, P(frame_col, a, stroke=8))
    c.drawLine(x + w * 0.8, y + h, x + w * 0.8, y + h + 60, P(frame_col, a, stroke=8))
    # words: (text, age, current)
    size = h * 0.5
    f = font(fnt, size)
    total = sum(f.measureText(wd + " ") for wd, _, _ in words)
    scale = min(1.0, (w - 40) / max(1, total))
    c.save()
    c.translate(x + w / 2, y + h / 2 + size * 0.35 * scale)
    c.scale(scale, scale)
    xx = -total / 2
    for wd, age, cur in words:
        ww = f.measureText(wd + " ")
        if age >= 0:
            on = clamp(age / 0.06)
            buzz = 0.75 + 0.25 * math.sin(t * 60 + xx) if age < 0.25 else 1.0
            cc = "#ffffff" if cur else color
            c.drawString(wd, xx, 0, f, P(color, 0.55 * a * on * buzz, stroke=size * 0.18, blur=size * 0.12))
            c.drawString(wd, xx, 0, f, P(mix(cc, "#ffffff", 0.35), a * on * buzz))
        else:
            c.drawString(wd, xx, 0, f, P("#3a2a50", 0.5 * a, stroke=1.5))
        xx += ww
    c.restore()


# ----------------------------------------------------------------------------
# practice room (cold open / ending)
# ----------------------------------------------------------------------------
def room(c, t, light=1.0, window_dim=1.0, sag=0.0, shake=(0.0, 0.0), amp_glow=0.0, fairy=1.0, trophy=True,
         bass=0.0, guitar_on_stand=True, poster_flap=0.0):
    c.drawRect(skia.Rect.MakeLTRB(-200, -200, W + 200, H + 200),
               P(shader=lin_grad((0, 0), (0, H), [(0, mix("#1b1233", "#2a1a4a", light), 1), (0.8, mix("#120b24", "#24163e", light), 1)])))
    # wallpaper stripes
    for i in range(22):
        x = i * 96
        c.drawRect(skia.Rect.MakeLTRB(x, -200, x + 44, 930), P("#ffffff", 0.025 * light))
    # window with the night city
    wx0, wy0, wx1, wy1 = 1160, 110, 1700, 600
    c.save()
    c.clipRect(skia.Rect.MakeLTRB(wx0, wy0, wx1, wy1))
    sky(c, [(0, "#0a0a2a"), (0.7, "#2a1a5a"), (1, "#6a2a70")], wy0, wy1)
    stars(c, t, 0.9 * window_dim, seed=77, n=120, h=wy1)
    moon(c, wx0 + 400, wy0 + 110, 45, a=window_dim)
    city(c, "far", 200, wy1 + 60, t, a=window_dim, scale=0.5)
    city(c, "mid", 900, wy1 + 80, t, a=window_dim, scale=0.6)
    c.restore()
    c.drawRect(skia.Rect.MakeLTRB(wx0, wy0, wx1, wy1), P("#0e0818", 1, stroke=26))
    c.drawLine((wx0 + wx1) / 2, wy0, (wx0 + wx1) / 2, wy1, P("#0e0818", 1, stroke=14))
    c.drawLine(wx0, (wy0 + wy1) / 2, wx1, (wy0 + wy1) / 2, P("#0e0818", 1, stroke=14))
    c.drawRect(skia.Rect.MakeLTRB(wx0 - 30, wy1 + 8, wx1 + 30, wy1 + 34), P("#3a2a50"))
    # posters (original art)
    _poster_bolt(c, 140, 120, 250, 340, t, poster_flap)
    _poster_cat(c, 440, 150, 220, 290, t, poster_flap)
    _poster_band(c, 770, 110, 240, 330, t, poster_flap)
    # fairy lights
    pth = skia.Path()
    pts = []
    for i in range(40):
        x = -40 + i * 52
        y = 60 + 36 * math.sin(i / 39 * math.pi * 4) ** 2 + sag * 60 * math.sin(i / 39 * math.pi)
        pts.append((x, y))
    c.drawPath(path_from(pts, close=False, smooth_k=0.6), P("#0b0714", 1, stroke=3))
    for i, (x, y) in enumerate(pts):
        cc = ["#ffd27a", "#ff7ac0", "#7af0ff", "#b8ff7a"][i % 4]
        on = fairy * (0.6 + 0.4 * math.sin(t * 3 + i * 1.3))
        c.drawCircle(x, y + 10, 7, P(cc, on))
        c.drawCircle(x, y + 10, 22, P(cc, 0.3 * on, blur=10))
    # shelf with trophy
    c.drawRect(skia.Rect.MakeLTRB(90, 560, 620, 585), P("#5a3a2a"))
    c.drawRect(skia.Rect.MakeLTRB(90, 585, 620, 596), P("#2a1a14"))
    for i, (bx, bw, bh, bc) in enumerate(((110, 34, 120, "#c0392b"), (146, 28, 105, "#2e86de"), (176, 38, 128, "#f2c14e"),
                                          (216, 26, 98, "#27ae60"))):
        c.drawRect(skia.Rect.MakeLTRB(bx, 560 - bh, bx + bw, 560), P(bc))
    c.drawRect(skia.Rect.MakeLTRB(540, 470, 600, 560), P("#7a4a2a"))
    c.drawCircle(570, 452, 40, P("#2f8a4a"))
    c.drawCircle(545, 470, 26, P("#3caa5a"))
    if trophy:
        K.draw_trophy(c, 400, 560, 150, t, shine=0.8)
    # amp stack (right)
    ax0, ay0 = 1480, 560
    c.drawRect(skia.Rect.MakeLTRB(ax0, ay0, ax0 + 360, ay0 + 420), P("#15111c"))
    c.drawRect(skia.Rect.MakeLTRB(ax0, ay0, ax0 + 360, ay0 + 420), P("#2c2638", 1, stroke=10))
    c.drawRect(skia.Rect.MakeLTRB(ax0 + 20, ay0 + 16, ax0 + 340, ay0 + 70), P("#f2c14e"))
    for k in range(6):
        kx = ax0 + 50 + k * 50
        c.drawCircle(kx, ay0 + 43, 14, P("#1a1520"))
        c.drawCircle(kx, ay0 + 43, 22, P("#ff4040", 0.5 * amp_glow, blur=8))
    for k in range(2):
        cx_, cy_ = ax0 + 95 + k * 170, ay0 + 230
        r = 70 * (1 + 0.06 * bass)
        c.drawCircle(cx_, cy_, 78, P("#0c0a12"))
        c.drawCircle(cx_, cy_, r, P(shader=rad_grad((cx_, cy_), r, [(0, "#4a4460", 1), (0.7, "#221e2c", 1), (1, "#110e16", 1)])))
        c.drawCircle(cx_, cy_ + 150, 60, P("#0c0a12"))
    c.drawLine(ax0 + 20, ay0 + 330, ax0 + 340, ay0 + 330, P("#2c2638", 1, stroke=6))
    glow(c, ax0 + 180, ay0 + 43, 260, "#ff5050", 0.35 * amp_glow)
    # guitar stand
    if guitar_on_stand:
        from . import hero as Z
        from .hero import Ink
        c.save()
        c.translate(1330, 900)
        c.rotate(-80)
        c.scale(620, 620)
        Z.draw_guitar(Ink(c, 0.012), 0.0)
        c.restore()
    c.drawLine(1300, 980, 1330, 880, P("#101014", 1, stroke=10))
    c.drawLine(1360, 980, 1330, 880, P("#101014", 1, stroke=10))
    # desk / floor
    c.drawRect(skia.Rect.MakeLTRB(-100, 930, W + 100, H + 200), P(shader=lin_grad((0, 930), (0, H), [(0, "#3a2418", 1), (1, "#1a0e08", 1)])))
    for i in range(12):
        c.drawLine(i * 180 - 40, 930, i * 180 - 140, H, P("#2a1810", 1, stroke=3))
    # LED strip glow along the wall bottom
    c.drawRect(skia.Rect.MakeLTRB(-100, 900, W + 100, 930), P(shader=lin_grad((0, 900), (0, 930), [(0, "#b04aff", 0), (1, "#b04aff", 0.5 * light)])))


def _poster(c, x, y, w, h, flap, t):
    c.save()
    ang = math.sin(t * 20) * 2.5 * flap
    c.translate(x + w / 2, y)
    c.rotate(ang)
    c.translate(-(x + w / 2), -y)
    c.drawRect(skia.Rect.MakeLTRB(x + 8, y + 10, x + w + 8, y + h + 10), P("#000000", 0.35))


def _poster_bolt(c, x, y, w, h, t, flap):
    _poster(c, x, y, w, h, flap, t)
    c.drawRect(skia.Rect.MakeLTRB(x, y, x + w, y + h), P("#1f1a3a"))
    c.drawRect(skia.Rect.MakeLTRB(x, y, x + w, y + h), P("#ffd21f", 1, stroke=6))
    b = poly([(x + w * 0.55, y + 40), (x + w * 0.25, y + h * 0.55), (x + w * 0.48, y + h * 0.55), (x + w * 0.35, y + h - 60),
              (x + w * 0.78, y + h * 0.4), (x + w * 0.54, y + h * 0.4), (x + w * 0.7, y + 40)])
    c.drawPath(b, P("#ffd21f"))
    f = font("BlackOpsOne.ttf", 40)
    tw = f.measureText("VOLTAGE")
    c.drawString("VOLTAGE", x + w / 2 - tw / 2, y + h - 18, f, P("#ff4f7a"))
    c.restore()


def _poster_cat(c, x, y, w, h, t, flap):
    _poster(c, x, y, w, h, flap, t)
    c.drawRect(skia.Rect.MakeLTRB(x, y, x + w, y + h), P("#0e2a4a"))
    for i in range(20):
        c.drawCircle(x + hash1(i) * w, y + hash1(i + 50) * h, 2, P("#ffffff", 0.8))
    c.drawCircle(x + w / 2, y + h * 0.45, 70, P("#ffb35a"))
    c.drawPath(poly([(x + w / 2 - 60, y + h * 0.45 - 40), (x + w / 2 - 40, y + h * 0.45 - 95), (x + w / 2 - 15, y + h * 0.45 - 60)]), P("#ffb35a"))
    c.drawPath(poly([(x + w / 2 + 60, y + h * 0.45 - 40), (x + w / 2 + 40, y + h * 0.45 - 95), (x + w / 2 + 15, y + h * 0.45 - 60)]), P("#ffb35a"))
    c.drawCircle(x + w / 2 - 25, y + h * 0.43, 8, P("#1a1020"))
    c.drawCircle(x + w / 2 + 25, y + h * 0.43, 8, P("#1a1020"))
    c.drawCircle(x + w / 2, y + h * 0.45, 88, P("#bfe6ff", 0.8, stroke=5))
    f = font("PressStart2P.ttf", 16)
    tw = f.measureText("SPACE CAT")
    c.drawString("SPACE CAT", x + w / 2 - tw / 2, y + h - 22, f, P("#ffffff"))
    c.restore()


def _poster_band(c, x, y, w, h, t, flap):
    _poster(c, x, y, w, h, flap, t)
    c.drawRect(skia.Rect.MakeLTRB(x, y, x + w, y + h), P("#e8e0d0"))
    c.drawRect(skia.Rect.MakeLTRB(x + 14, y + 14, x + w - 14, y + h * 0.62), P("#e0243a"))
    for i in range(5):
        c.drawCircle(x + w / 2, y + h * 0.33, 20 + i * 18, P("#1a1020", 1, stroke=5))
    f = font("PermanentMarker.ttf", 44)
    tw = f.measureText("THE FUZZ")
    c.drawString("THE FUZZ", x + w / 2 - tw / 2, y + h * 0.78, f, P("#1a1020"))
    f2 = font("PatrickHand.ttf", 22)
    tw2 = f2.measureText("live - one night only")
    c.drawString("live - one night only", x + w / 2 - tw2 / 2, y + h * 0.9, f2, P("#1a1020"))
    c.restore()


def webcam_ui(c, t, chat, rec_t=None, alpha=1.0, title="200 SUB SPECIAL"):
    """Stream overlay: REC/LIVE badge, timer, title card and a scrolling chat panel."""
    a = alpha
    rt = t if rec_t is None else rec_t
    # corner brackets
    for (x, y, sx, sy) in ((40, 40, 1, 1), (W - 40, 40, -1, 1), (40, H - 40, 1, -1), (W - 40, H - 40, -1, -1)):
        c.drawLine(x, y, x + 60 * sx, y, P("#ffffff", 0.8 * a, stroke=4))
        c.drawLine(x, y, x, y + 60 * sy, P("#ffffff", 0.8 * a, stroke=4))
    on = 1.0 if (t % 1.0) < 0.6 else 0.25
    c.drawCircle(92, 98, 14, P("#ff2a3a", on * a))
    f = font("PressStart2P.ttf", 22)
    c.drawString("REC", 118, 108, f, P("#ffffff", a))
    mm, ss = int(rt // 60), int(rt % 60)
    c.drawString(f"00:{mm:02d}:{ss:02d}", 118, 142, font("PressStart2P.ttf", 16), P("#ffffff", 0.85 * a))
    c.drawRoundRect(skia.Rect.MakeLTRB(W - 250, 76, W - 90, 122), 8, 8, P("#ff2a3a", a))
    fl = font("Bangers.ttf", 36)
    c.drawString("LIVE", W - 222, 112, fl, P("#ffffff", a))
    c.drawString("● 200", W - 168 + 12, 112, font("Bangers.ttf", 30), P("#ffffff", 0.0))
    # title strip
    ft = font("Bangers.ttf", 46)
    tw = ft.measureText(title)
    c.drawRoundRect(skia.Rect.MakeLTRB(W / 2 - tw / 2 - 30, 60, W / 2 + tw / 2 + 30, 122), 14, 14, P("#120a20", 0.75 * a))
    c.drawString(title, W / 2 - tw / 2, 108, ft, P("#ffd21f", a))
    # chat panel
    px0, py0, px1, py1 = W - 520, 170, W - 60, 720
    c.drawRoundRect(skia.Rect.MakeLTRB(px0, py0, px1, py1), 16, 16, P("#0b0716", 0.62 * a))
    c.drawString("LIVE CHAT", px0 + 20, py0 + 38, font("Bangers.ttf", 30), P("#ffffff", 0.9 * a))
    visible = [m for m in chat if m[0] <= t]
    fu = font("PressStart2P.ttf", 13)
    fm = font("PatrickHand.ttf", 30)
    y = py1 - 24
    for (mt, user, msg, colr) in reversed(visible[-9:]):
        age = t - mt
        slide = ease_out(clamp(age / 0.25))
        aa = a * slide
        c.drawString(user, px0 + 20, y - 26, fu, P(colr, aa))
        c.drawString(msg, px0 + 20 + (1 - slide) * 40, y + 4, fm, P("#ffffff", aa))
        y -= 62
        if y < py0 + 70:
            break


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def train(c, scroll, roof_y, t, led_words=None, led_car=1, n_cars=4, car_w=1150, a=1.0, lights=1.0, tunnel=False,
          track=None):
    body_h = 300
    ts = scroll if track is None else track
    y0 = roof_y
    for k in range(n_cars):
        X0 = k * (car_w + 40) - scroll
        X1 = X0 + car_w
        if X1 < -100 or X0 > W + 100:
            continue
        body = rrect(X0, y0, X1, y0 + body_h, 26)
        c.drawPath(body, P(shader=lin_grad((0, y0), (0, y0 + body_h), [(0, "#c9d2e6", a), (0.5, "#8e9ab8", a), (1, "#5a6482", a)])))
        c.drawRect(skia.Rect.MakeLTRB(X0, y0 + 190, X1, y0 + 214), P("#ff3d8a", a))
        c.drawRect(skia.Rect.MakeLTRB(X0, y0 + 214, X1, y0 + 224), P("#2fe3d7", a))
        # windows with passengers
        for j in range(7):
            wx = X0 + 50 + j * 150
            if wx + 110 > X1 - 30:
                break
            win = rrect(wx, y0 + 50, wx + 110, y0 + 160, 12)
            c.drawPath(win, P("#1a2240", a))
            c.drawPath(win, P(shader=lin_grad((0, y0 + 50), (0, y0 + 160), [(0, "#ffe9a8", 0.85 * lights * a), (1, "#ffb35a", 0.6 * lights * a)])))
            for q in range(2):
                if hash1(k * 31 + j * 7 + q) < 0.6:
                    px = wx + 25 + q * 50
                    c.drawCircle(px, y0 + 108, 14, P("#2a1a30", 0.9 * a))
                    c.drawRect(skia.Rect.MakeLTRB(px - 18, y0 + 122, px + 18, y0 + 160), P("#2a1a30", 0.9 * a))
            c.drawPath(win, P("#3a4460", a, stroke=6))
        # doors
        for j in (0.33, 0.8):
            dx = X0 + car_w * j
            c.drawRect(skia.Rect.MakeLTRB(dx - 4, y0 + 40, dx + 4, y0 + body_h - 20), P("#4a5472", a))
        # LED destination board
        bx0, by0 = X0 + car_w * 0.5 - 300, y0 + 236
        c.drawRoundRect(skia.Rect.MakeLTRB(bx0, by0, bx0 + 600, by0 + 52), 6, 6, P("#0b0a0e", a))
        if led_words is not None and k == led_car:
            fled = font("VT323.ttf", 46)
            xx = bx0 + 16
            for wd, age, cur in led_words:
                if age < 0:
                    continue
                cc = "#ffb000" if not cur else "#fff2a0"
                c.drawString(wd, xx, by0 + 42, fled, P(cc, a))
                c.drawString(wd, xx, by0 + 42, fled, P("#ff9000", 0.4 * a, blur=5))
                xx += fled.measureText(wd + " ")
                if xx > bx0 + 590:
                    break
        else:
            fled = font("VT323.ttf", 40)
            msg = "LOCAL  ▶  DOWNTOWN" if k % 2 == 0 else "NEXT: ??? "
            c.drawString(msg, bx0 + 16, by0 + 40, fled, P("#ff9000", 0.8 * a))
        # roof details
        c.drawRect(skia.Rect.MakeLTRB(X0 + 80, y0 - 18, X0 + 220, y0 + 2), P("#6a7490", a))
        c.drawRect(skia.Rect.MakeLTRB(X1 - 220, y0 - 18, X1 - 80, y0 + 2), P("#6a7490", a))
        if k % 2 == 0:
            px = X0 + car_w * 0.5
            c.drawLine(px - 60, y0, px, y0 - 70, P("#2a2a36", a, stroke=6))
            c.drawLine(px + 60, y0, px, y0 - 70, P("#2a2a36", a, stroke=6))
            c.drawLine(px - 70, y0 - 72, px + 70, y0 - 72, P("#2a2a36", a, stroke=6))
        # bogies / wheels
        for wxk in (0.15, 0.3, 0.7, 0.85):
            wx = X0 + car_w * wxk
            c.drawCircle(wx, y0 + body_h + 20, 28, P("#1a1a22", a))
            ang = ts * 0.05
            c.drawLine(wx, y0 + body_h + 20, wx + math.cos(ang) * 20, y0 + body_h + 20 + math.sin(ang) * 20, P("#6a6a7a", a, stroke=5))
        c.drawRect(skia.Rect.MakeLTRB(X1 - 2, y0 + 60, X1 + 42, y0 + 240), P("#2a2a36", a))
    # rails
    c.drawRect(skia.Rect.MakeLTRB(-50, y0 + body_h + 44, W + 50, y0 + body_h + 56), P("#6a6a7a", a))
    c.drawRect(skia.Rect.MakeLTRB(-50, y0 + body_h + 56, W + 50, H + 300), P("#120c1c", a))
    for i in range(30):
        x = (i * 90 - ts) % (W + 180) - 90
        c.drawRect(skia.Rect.MakeLTRB(x, y0 + body_h + 56, x + 30, y0 + body_h + 70), P("#2a2030", a))


def catenary(c, scroll, top_y, t, a=1.0, spacing=900):
    for i in range(-1, int(W / spacing) + 3):
        x = i * spacing - (scroll % spacing)
        c.drawRect(skia.Rect.MakeLTRB(x - 10, top_y - 40, x + 10, H + 100), P("#0c0a14", a))
        c.drawLine(x - 10, top_y, x + 140, top_y, P("#0c0a14", a, stroke=8))
    for k, dy in enumerate((0, 26)):
        pth = skia.Path()
        for i in range(-1, int(W / spacing) + 3):
            x = i * spacing - (scroll % spacing)
            if i == -1:
                pth.moveTo(x, top_y + dy)
            pth.quadTo(x + spacing / 2, top_y + dy + 40, x + spacing, top_y + dy)
        c.drawPath(pth, P("#0c0a14", a, stroke=3 - k))


def tunnel(c, scroll, t, strobe=0.0, words=None, word_style="graffiti", a=1.0):
    c.drawRect(skia.Rect.MakeLTRB(-50, -50, W + 50, H + 50), P("#07050c"))
    c.drawRect(skia.Rect.MakeLTRB(-50, 60, W + 50, 700), P(shader=lin_grad((0, 60), (0, 700), [(0, "#15101f", 1), (1, "#0a0712", 1)])))
    # wall ribs
    for i in range(12):
        x = (i * 260 - scroll * 1.0) % (12 * 260) - 260
        c.drawRect(skia.Rect.MakeLTRB(x, 60, x + 40, 700), P("#1d1628"))
    # passing lights
    for i in range(8):
        x = (i * 420 - scroll * 1.0) % (8 * 420) - 420
        c.drawRoundRect(skia.Rect.MakeLTRB(x, 90, x + 120, 110), 8, 8, P("#fff3c0", 0.9))
        c.drawPath(poly([(x - 60, 110), (x + 180, 110), (x + 320, 700), (x - 200, 700)]), P("#fff0b0", 0.07 + 0.08 * strobe))
    # graffiti words on the wall
    if words:
        f = font("PermanentMarker.ttf", 96)
        xx = 200 - 0 * scroll
        for i, (wd, age, cur) in enumerate(words):
            if age < 0:
                continue
            row = i // 3
            wx = 900 + (i % 3) * 330 + row * 120
            reveal = clamp(age / 0.15)
            cc = ["#ff4fb0", "#2fe3d7", "#ffd21f", "#9aff7a"][i % 4]
            c.save()
            c.translate(wx + math.sin(i) * 20, 200 + row * 150)
            c.rotate(-6 + (i % 3) * 5)
            c.scale(0.6 + 0.4 * reveal, 0.6 + 0.4 * reveal)
            c.drawString(wd, 0, 0, f, P("#000000", 0.6 * reveal, stroke=26))
            c.drawString(wd, 0, 0, f, P(cc, reveal))
            c.drawString(wd, 0, 0, f, P("#ffffff", 0.5 * reveal, stroke=3))
            c.restore()


# ----------------------------------------------------------------------------
# radio tower (solo stage) and the kaiju blob city
# ----------------------------------------------------------------------------
def radio_tower(c, x, base_y, top_y, t, a=1.0, color="#16102c"):
    w0, w1 = 180, 26
    c.drawLine(x - w0, base_y, x - w1, top_y, P("#7afcff", 0.35 * a, stroke=14, blur=6))
    c.drawLine(x + w0, base_y, x + w1, top_y, P("#7afcff", 0.35 * a, stroke=14, blur=6))
    lp = P(color, a, stroke=7)
    c.drawLine(x - w0, base_y, x - w1, top_y, lp)
    c.drawLine(x + w0, base_y, x + w1, top_y, lp)
    n = 12
    for i in range(n):
        f0, f1 = i / n, (i + 1) / n
        y0, y1 = lerp(base_y, top_y, f0), lerp(base_y, top_y, f1)
        xa, xb = lerp(w0, w1, f0), lerp(w0, w1, f1)
        c.drawLine(x - xa, y0, x + xb, y1, P(color, a, stroke=4))
        c.drawLine(x + xa, y0, x - xb, y1, P(color, a, stroke=4))
        c.drawLine(x - xb, y1, x + xb, y1, P(color, a, stroke=4))
    c.drawRect(skia.Rect.MakeLTRB(x - 60, top_y - 16, x + 60, top_y + 4), P(color, a))
    on = 0.5 + 0.5 * math.sin(t * 5)
    c.drawCircle(x, top_y - 120, 7, P("#ff3355", a * on))
    c.drawLine(x, top_y - 16, x, top_y - 120, P(color, a, stroke=6))


# ----------------------------------------------------------------------------
# inside the blob
# ----------------------------------------------------------------------------
def blob_dimension(c, t, b, pulse=0.0):
    c.drawRect(skia.Rect.MakeLTRB(-100, -100, W + 100, H + 100),
               P(shader=rad_grad((W / 2, H / 2), W * 0.75, [(0, "#ffd0ea", 1), (0.45, "#ff79bd", 1), (1, "#8a1a5c", 1)])))
    for i in range(10):
        r = ((b * 0.25 + i / 10) % 1.0) * 1300
        pts = []
        for k in range(48):
            ang = k / 48 * TAU
            rr = r * (1 + 0.06 * math.sin(5 * ang + t * 2 + i) + 0.04 * math.sin(3 * ang - t * 1.3))
            pts.append((W / 2 + math.cos(ang) * rr, H / 2 + math.sin(ang) * rr * 0.75))
        c.drawPath(path_from(pts, close=True, smooth_k=1.0), P("#ffffff", 0.25 * (1 - r / 1300), stroke=6))
    rng = np.random.default_rng(3)
    for i in range(40):
        x0 = rng.uniform(0, W)
        sp = rng.uniform(40, 120)
        y = H + 60 - ((t * sp + rng.uniform(0, H)) % (H + 120))
        r = rng.uniform(6, 26)
        x = x0 + math.sin(t + i) * 30
        c.drawCircle(x, y, r, P("#ffffff", 0.45, stroke=2.5))
        c.drawCircle(x - r * 0.35, y - r * 0.35, r * 0.2, P("#ffffff", 0.6))


def floaty_items(c, t, a=1.0, vhs_focus=0.0):
    """Original embarrassing keepsakes drifting around the cringe dimension."""
    items = [("vhs", 380, 300, 1.0), ("duck", 1500, 260, 0.8), ("heart", 260, 780, 0.9), ("phone", 1650, 760, 0.9),
             ("diary", 980, 170, 0.8), ("glitter", 1280, 900, 0.8)]
    for i, (kind, x, y, s) in enumerate(items):
        x += math.sin(t * 0.6 + i) * 40
        y += math.cos(t * 0.5 + i * 2) * 30
        ang = math.sin(t * 0.4 + i) * 20
        c.save()
        c.translate(x, y)
        c.rotate(ang)
        k = s * (1 + (0.6 * vhs_focus if kind == "vhs" else 0))
        c.scale(k, k)
        if kind == "vhs":
            c.drawRoundRect(skia.Rect.MakeLTRB(-130, -75, 130, 75), 10, 10, P("#1a1420", a))
            c.drawRect(skia.Rect.MakeLTRB(-110, -60, 110, 5), P("#f4f0e0", a))
            c.drawString("MY FIRST VIDEO", -100, -30, font("PermanentMarker.ttf", 26), P("#1a1020", a))
            c.drawString("3 views", -100, -4, font("PatrickHand.ttf", 26), P("#c0145a", a))
            for sd in (-1, 1):
                c.drawCircle(sd * 55, 40, 22, P("#3a3440", a))
                c.drawCircle(sd * 55, 40, 8, P("#1a1420", a))
        elif kind == "duck":
            c.drawCircle(0, 10, 50, P("#ffd21f", a))
            c.drawCircle(30, -40, 32, P("#ffd21f", a))
            c.drawPath(poly([(55, -40), (90, -32), (55, -24)]), P("#ff8a1f", a))
            c.drawCircle(38, -48, 6, P("#1a1020", a))
        elif kind == "heart":
            h = skia.Path()
            h.moveTo(0, 40)
            h.cubicTo(-80, -10, -40, -70, 0, -30)
            h.cubicTo(40, -70, 80, -10, 0, 40)
            c.drawPath(h, P("#ff3d7a", a))
            c.drawPath(h, P("#ffffff", 0.8 * a, stroke=5))
        elif kind == "phone":
            c.drawRoundRect(skia.Rect.MakeLTRB(-40, -80, 40, 80), 14, 14, P("#c0c8d8", a))
            c.drawRect(skia.Rect.MakeLTRB(-30, -65, 30, -10), P("#7ae0ff", a))
            for r_ in range(3):
                for q in range(3):
                    c.drawCircle(-20 + q * 20, 12 + r_ * 20, 6, P("#6a7488", a))
        elif kind == "diary":
            c.drawRoundRect(skia.Rect.MakeLTRB(-70, -90, 70, 90), 8, 8, P("#b44aff", a))
            c.drawString("DIARY", -52, -20, font("Bangers.ttf", 40), P("#ffffff", a))
            c.drawRect(skia.Rect.MakeLTRB(55, -10, 80, 20), P("#ffd21f", a))
            K.sparkle(c, 0, 40, 26, "#ffffff", a)
        elif kind == "glitter":
            c.drawRoundRect(skia.Rect.MakeLTRB(-16, -90, 16, 90), 10, 10, P("#ff7ac0", a))
            for q in range(6):
                K.sparkle(c, math.sin(q * 2 + t * 3) * 40, -60 + q * 24, 10, "#ffffff", a)
        c.restore()
