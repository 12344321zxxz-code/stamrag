"""Scene changes: the title card sliding off like a sheet, and foreground wipes (a clump of grass,
big leaves, reeds or a wave sweeping past the lens, blurred as if very close)."""
import math
from functools import lru_cache

import numpy as np
import skia

from . import world as Wd
from .craft import TAU, Layer, col, cut, piece, smoother, torn_piece
from .core import H, W

WIPE_W = 2700


def _sheet_outline(seed):
    """A tall sheet with big, irregular torn left and right edges."""
    ys = np.linspace(-150, H + 150, 60)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, TAU, 4)
    def edge(x0, k):
        return x0 + 55 * np.sin(ys / 260 + ph[k]) + 25 * np.sin(ys / 97 + ph[k + 1])
    left = np.stack([edge(230, 0), ys], 1)
    right = np.stack([edge(WIPE_W - 230, 2), ys[::-1]], 1)
    return np.vstack([left, right])


@lru_cache(None)
def _wipe_layer(kind):
    def draw(c):
        rng = np.random.default_rng(abs(hash(kind)) % 2 ** 31)
        outline = _sheet_outline(len(kind))
        if kind.startswith("wave"):
            base = {"wave": "sea2", "wave2": "sea", "wave3": "sea3"}[kind]
        else:
            greens = {"grass": ["#5c9e3d", "#4f8f35", "#74b64f"], "leaves": ["#3d7f31", "#4f9a3f", "#5fae4a"],
                      "reeds": ["#6f9e3f", "#5d8a33", "#86b24f"], "leaves2": ["#2f7a3c", "#4f9a3f", "#6cc04a"]}[kind]
            base = greens[1]
        inner = torn_piece(c, outline, base, 7, lift=3.0, rim_w=7.0, amp=4.0, grad=0.0)
        c.save()
        c.clipPath(inner, doAntiAlias=True)
        if kind.startswith("wave"):
            for k in range(8):
                y = -80 + k * 170
                pts = Wd.wave_band(0, WIPE_W, y, 38, 380, 30 + k, depth=260)
                torn_piece(c, pts, ["sea4", "sea", "sea2", "sea3"][k % 4], 40 + k, lift=2.5, amp=3.0)
        else:
            for k in range(46):
                x = rng.uniform(150, WIPE_W - 150)
                colr = greens[k % 3]
                if kind in ("leaves", "leaves2"):
                    Wd.leaf(c, x, rng.uniform(-100, H + 100), rng.uniform(380, 620), rng.uniform(90, 150),
                            rng.uniform(0, 360), colr, 60 + k, lift=3.0)
                else:
                    h = rng.uniform(900, 1400)
                    w = rng.uniform(60, 140) if kind == "grass" else rng.uniform(30, 60)
                    lean = rng.uniform(-18, 18)
                    a = math.radians(lean - 90)
                    tip = (x + math.cos(a) * h, H + 120 + math.sin(a) * h)
                    pts = [(x - w, H + 140), (tip[0] - w * 0.2, tip[1] + 40), tip, (tip[0] + w * 0.2, tip[1] + 40),
                           (x + w, H + 140)]
                    piece(c, cut(pts, 60 + k, 1.5), colr, 60 + k, lift=3.0, grad=0.12)
        c.restore()
    return Layer((0, -160, WIPE_W, H + 160), draw, blur=2.5, res=0.5)


def wipe(c, kind, u):
    """u 0..1 across the window; the element fully covers the frame at u = 0.5."""
    L = _wipe_layer(kind)
    x_center = (W + WIPE_W / 2 - 150) * (1 - u) + (150 - WIPE_W / 2) * u
    c.save()
    c.resetMatrix()
    L.draw(c, dx=x_center - WIPE_W / 2)
    c.restore()


def sheet_slide(c, draw_fn, u):
    """The title card is a sheet on top of the next scene; it slides off to the left."""
    if u >= 1:
        return
    x = -W * 1.08 * smoother(u) ** 1.3
    rot = -3.0 * smoother(u)
    c.save()
    c.resetMatrix()
    c.translate(x, 0)
    c.rotate(rot)
    edge = skia.Paint(AntiAlias=True, Color=col("shadow", 0.35),
                      MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 18))
    c.drawRect(skia.Rect.MakeLTRB(W - 30, -40, W + 22, H + 40), edge)
    c.clipRect(skia.Rect.MakeWH(W, H))
    draw_fn(c)
    c.restore()
