"""The cast, as paper puppets drawn in their own local coordinates (x right, y down, origin at the
character's anchor: the waterline for floaters, the feet for sitters).

Every function takes the canvas, a position/scale and a small pose dict, so scenes only animate
numbers. Faces share one set of helpers (cut-paper eyes, felt-pen mouths, blush).
"""
import math

import numpy as np
import skia

from .craft import TAU, brad, clamp, col, cpath, cut, drop, ellipse, lerp, paper_paint, path_of, piece, smooth_path, stroke


# ----------------------------------------------------------------------------------------------------
# faces
# ----------------------------------------------------------------------------------------------------
def eye(c, x, y, r, look=(0.0, 0.0), blink=0.0, mood="happy", seed=0, lid_col=None, lift=0.5):
    """Cut-paper eye: white disc, black pupil, highlight. blink 0..1 squashes it shut."""
    lx, ly = look
    if mood == "joy" or blink > 0.85:
        # closed happy arc
        k = 1 if mood == "joy" else -1
        pts = [(x - r * 0.9, y + r * 0.15 * k), (x, y - r * 0.55 * k), (x + r * 0.9, y + r * 0.15 * k)]
        stroke(c, pts, max(2.5, r * 0.28), "ink")
        return
    sy = 1.0 - 0.9 * clamp(blink)
    c.save()
    c.translate(x, y)
    c.scale(1.0, sy)
    white = path_of(ellipse(0, 0, r * 0.86, r, 40))
    piece(c, white, "white", seed, lift=lift, rim=0, grad=0.04, shadow=0.22)
    pr = r * 0.58
    px, py = lx * r * 0.32, ly * r * 0.32 + r * 0.08
    c.drawCircle(px, py, pr, skia.Paint(AntiAlias=True, Color=col("#1d1a1a")))
    c.drawCircle(px + pr * 0.38, py - pr * 0.42, pr * 0.34, skia.Paint(AntiAlias=True, Color=col("white")))
    c.drawCircle(px - pr * 0.35, py + pr * 0.4, pr * 0.14, skia.Paint(AntiAlias=True, Color=col("white", 0.8)))
    if mood == "sleepy" and lid_col is not None:
        lid = skia.Path()
        lid.moveTo(-r * 1.2, -r * 1.3)
        lid.lineTo(r * 1.2, -r * 1.3)
        lid.lineTo(r * 1.2, -r * 0.55)
        lid.lineTo(-r * 1.2, -r * 0.05)
        lid.close()
        c.save()
        c.clipPath(white, doAntiAlias=True)
        c.drawPath(lid, paper_paint(lid_col, seed + 9))
        c.restore()
    c.restore()


def brow(c, x, y, w, tilt, width=3.5):
    stroke(c, [(x - w / 2, y + tilt), (x, y - 2), (x + w / 2, y - tilt)], width, "ink")


def mouth(c, x, y, w, open_=0.0, smile=1.0, shape="smile"):
    """Felt-pen smile that opens into a cut-paper mouth when talking."""
    if shape == "o":
        h = w * (0.45 + 0.35 * open_)
        p = path_of(ellipse(x, y + h * 0.2, w * 0.28, h * 0.5, 30))
        c.drawPath(p, skia.Paint(AntiAlias=True, Color=col("#6b1b28")))
        return
    if open_ < 0.08:
        pts = [(x - w / 2, y - 2 * smile), (x - w * 0.2, y + w * 0.18 * smile), (x + w * 0.2, y + w * 0.18 * smile),
               (x + w / 2, y - 2 * smile)]
        stroke(c, pts, max(2.5, w * 0.1), "ink")
        return
    h = w * (0.18 + 0.55 * clamp(open_))
    p = skia.Path()
    p.moveTo(x - w / 2, y - 2)
    p.cubicTo(x - w * 0.25, y + 1, x + w * 0.25, y + 1, x + w / 2, y - 2)
    p.cubicTo(x + w * 0.42, y + h, x - w * 0.42, y + h, x - w / 2, y - 2)
    p.close()
    c.drawPath(p, skia.Paint(AntiAlias=True, Color=col("#6b1b28")))
    c.save()
    c.clipPath(p, doAntiAlias=True)
    c.drawOval(skia.Rect.MakeXYWH(x - w * 0.28, y + h * 0.45, w * 0.56, h * 0.7), skia.Paint(AntiAlias=True,
                                                                                            Color=col("#f06a7f")))
    c.restore()
    c.drawPath(p, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=max(2.0, w * 0.07),
                             Color=col("ink"), StrokeJoin=skia.Paint.kRound_Join))


def cheeks(c, x, y, dx, r, alpha=0.55):
    p = skia.Paint(AntiAlias=True, Color=col("cheek", alpha))
    p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, r * 0.25))
    c.drawOval(skia.Rect.MakeXYWH(x - dx - r, y - r * 0.6, 2 * r, 1.2 * r), p)
    c.drawOval(skia.Rect.MakeXYWH(x + dx - r, y - r * 0.6, 2 * r, 1.2 * r), p)


# ----------------------------------------------------------------------------------------------------
# Pip, the paper boat
# ----------------------------------------------------------------------------------------------------
PIP_HULL = [(-165, -26), (165, -26), (112, 50), (-112, 50)]
PIP_SAIL = [(-84, -26), (0, -178), (84, -26)]
MAST_TOP = (0, -276)
RIBBON_SLOTS = [-266, -251, -236, -221, -206, -191]


def _ruled(c, clip, seed, x0=-170, x1=170, y0=-190, y1=60, margin=True):
    c.save()
    c.clipPath(clip, doAntiAlias=True)
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.6, Color=col("#8fb9e3", 0.55))
    y = y0 + 7
    while y < y1:
        c.drawLine(x0, y, x1, y, p)
        y += 15
    if margin:
        c.drawLine(-128, y0, -128, y1, skia.Paint(AntiAlias=True, StrokeWidth=1.8, Color=col("#ef7d8d", 0.55)))
    c.restore()


def ribbon_path(x0, y0, length, width, t, phase, calm=0.0, droop=0.0):
    """A streamer tied at (x0, y0) flowing to the left, rippling."""
    n = 18
    top, bot = [], []
    for i in range(n + 1):
        u = i / n
        x = x0 - length * u
        amp = (5 + 9 * u) * (1 - 0.6 * calm)
        y = y0 + amp * math.sin(TAU * (u * 1.3 - t * 1.6) + phase) * u + droop * u * u * length
        w = width * (1 - 0.25 * u)
        top.append((x, y - w / 2))
        bot.append((x, y + w / 2))
    tip = (x0 - length - width * 0.7, top[-1][1] + width * 0.5)
    return path_of(top + [tip] + bot[::-1])


def pip_point(x, y, s, rot, lx, ly):
    """Local Pip point -> world."""
    a = math.radians(rot)
    return x + s * (lx * math.cos(a) - ly * math.sin(a)), y + s * (lx * math.sin(a) + ly * math.cos(a))


def pip(c, x, y, s=1.0, rot=0.0, t=0.0, mouth_open=0.0, look=(0.0, 0.0), blink=0.0, mood="happy",
        ribbons=(), mast=True, squash=0.0, shadow_lift=1.5):
    """Pip at waterline (x, y). ribbons: list of (colour, amount 0..1 attached)."""
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s * (1 + 0.12 * squash), s * (1 - 0.12 * squash))
    seedb = 101
    # mast (toothpick) + ribbons behind the sail
    if mast:
        mp = path_of([(-3.2, -150), (-2.6, -268), (0, -282), (2.6, -268), (3.2, -150)])
        piece(c, mp, "#d9b27c", 7, lift=1.0, rim=0.2, grad=0.15)
        for i, (colr, amt) in enumerate(ribbons):
            if amt <= 0:
                continue
            ln = 120 * amt
            rp = ribbon_path(0, RIBBON_SLOTS[i], ln, 11.5, t, i * 0.9)
            piece(c, rp, colr, 200 + i, lift=1.2, rim=0.25, grad=0.1, shadow=0.25)
    sail = cpath(PIP_SAIL, seedb + 1, 0.8)
    hull = cpath(PIP_HULL, seedb + 2, 0.8)
    # sail: two facets split by the centre crease
    drop(c, sail, shadow_lift, 0.3)
    c.drawPath(sail, paper_paint("#fbf8ef", seedb + 3))
    c.save()
    c.clipPath(sail, doAntiAlias=True)
    right = path_of([(0, -190), (100, -20), (0, -20)])
    c.drawPath(right, skia.Paint(AntiAlias=True, Color=col("#c9c2b3", 0.35)))
    c.restore()
    _ruled(c, sail, seedb, margin=False)
    c.drawLine(0, -176, 0, -27, skia.Paint(AntiAlias=True, StrokeWidth=1.2, Color=col("#b8ae9c", 0.8)))
    c.drawPath(sail, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.4,
                                Color=col("#ffffff", 0.5)))
    # hull: three facets (left flap, middle, right flap)
    drop(c, hull, shadow_lift, 0.32)
    c.drawPath(hull, paper_paint("#fbf8ef", seedb + 4))
    c.save()
    c.clipPath(hull, doAntiAlias=True)
    c.drawPath(path_of([(-170, -30), (-84, -26), (-112, 55), (-170, 55)]),
               skia.Paint(AntiAlias=True, Color=col("#ffffff", 0.45)))
    c.drawPath(path_of([(170, -30), (84, -26), (112, 55), (170, 55)]),
               skia.Paint(AntiAlias=True, Color=col("#b9b09e", 0.38)))
    c.restore()
    _ruled(c, hull, seedb)
    crease = skia.Paint(AntiAlias=True, StrokeWidth=1.3, Color=col("#aa9f8b", 0.75))
    c.drawLine(-84, -26, -112, 50, crease)
    c.drawLine(84, -26, 112, 50, crease)
    c.drawLine(-165, -26, 165, -26, skia.Paint(AntiAlias=True, StrokeWidth=1.2, Color=col("#b3a893", 0.7)))
    c.drawPath(hull, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.4,
                                Color=col("#ffffff", 0.5)))
    # face
    cheeks(c, 0, 12, 58, 13, 0.5)
    lid = "#f3eee2"
    if mood == "sad":
        brow(c, -34, -29, 26, 5, 3.2)
        brow(c, 34, -29, 26, -5, 3.2)
    elif mood in ("excited", "wow"):
        brow(c, -34, -34, 24, 2, 3.2)
        brow(c, 34, -34, 24, -2, 3.2)
    eye(c, -33, -6, 15, look, blink, "joy" if mood == "joy" else mood, seedb + 5, lid)
    eye(c, 33, -6, 15, look, blink, "joy" if mood == "joy" else mood, seedb + 6, lid)
    if mood == "wow" and mouth_open < 0.1:
        mouth(c, 0, 17, 26, 0.6, shape="o")
    else:
        mouth(c, 0, 18, 30, mouth_open, smile=-0.6 if mood == "sad" else 1.0)
    c.restore()


# ----------------------------------------------------------------------------------------------------
# Lulu the ladybug (faces left). anchor: middle of her feet
# ----------------------------------------------------------------------------------------------------
def lulu(c, x, y, s=1.0, rot=0.0, t=0.0, mouth_open=0.0, blink=0.0, wings=0.0, look=(-0.4, 0.0), legs=0.0):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s, s)
    # legs
    for i, lx in enumerate((-30, 0, 30)):
        a = math.sin(t * 20 + i) * 6 * legs
        stroke(c, [(lx, -12), (lx - 10 + a, 4), (lx - 16 + a, 8)], 4.5, "ink")
    # wings (vellum) under the shell
    if wings > 0.01:
        flap = 0.55 + 0.45 * math.sin(t * 55)
        for side in (-1, 1):
            c.save()
            c.translate(8, -52)
            c.rotate(side * (35 + 25 * flap) * wings - 10)
            wp = path_of(ellipse(18 * side + 30, -38, 34 * wings, 70 * wings, 36))
            c.drawPath(wp, skia.Paint(AntiAlias=True, Color=col("#eef6fb", 0.72)))
            c.drawPath(wp, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.2,
                                      Color=col("#9fb3c2", 0.8)))
            c.restore()
    # shell: lifts at the back when she flies
    c.save()
    c.translate(-30, -34)
    c.rotate(-22 * wings)
    c.translate(30, 34)
    dome = [(4 + 66 * math.cos(a), -32 + 60 * math.sin(a)) for a in np.linspace(math.pi, TAU, 34)]
    dome += [(64, -24), (-56, -24)]
    shell = cpath(dome, 41, 0.8)
    piece(c, shell, "red", 41, lift=1.6, rim=0.3, grad=0.14)
    for dx, dy, r in ((-22, -60, 9), (12, -74, 9), (40, -52, 8), (0, -40, 7), (-36, -36, 6), (30, -30, 5)):
        c.drawCircle(dx, dy, r, skia.Paint(AntiAlias=True, Color=col("#2b2727")))
    c.restore()
    # face on the head
    # head (black) with antennae
    for side in (-1, 1):
        ax = -84 + side * 10
        stroke(c, [(ax, -64), (ax - 8 + side * 4, -92), (ax - 16 + side * 8, -104)], 3.5, "ink")
        c.drawCircle(ax - 16 + side * 8, -104, 5.5, skia.Paint(AntiAlias=True, Color=col("ink")))
    head = cpath(ellipse(-80, -40, 36, 33, 40), 31, 0.7)
    piece(c, head, "#2b2727", 31, lift=1.0, rim=0.25)
    eye(c, -94, -46, 11, look, blink, "happy", 33, lift=0.3)
    eye(c, -70, -46, 11, look, blink, "happy", 34, lift=0.3)
    cheeks(c, -82, -30, 18, 6, 0.65)
    mouth(c, -82, -24, 15, mouth_open)
    c.restore()


# ----------------------------------------------------------------------------------------------------
# Finn the goldfish (faces left). anchor: body centre
# ----------------------------------------------------------------------------------------------------
def finn(c, x, y, s=1.0, rot=0.0, t=0.0, mouth_open=0.0, blink=0.0, wag=0.0, flip=False):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(-s if flip else s, s)
    # tail on a split pin
    c.save()
    c.translate(58, 0)
    c.rotate(math.sin(t * 9) * (10 + 18 * wag))
    tail = cpath([(0, 0), (60, -52), (78, -40), (56, 0), (78, 40), (60, 52)], 51, 0.8)
    piece(c, tail, "#ffb347", 51, lift=1.2, grad=0.1)
    for a in (-0.55, -0.25, 0.25, 0.55):
        c.drawLine(8, 0, 8 + 58 * math.cos(a), 58 * math.sin(a), skia.Paint(AntiAlias=True, StrokeWidth=1.6,
                                                                        Color=col("#e07a12", 0.6)))
    c.restore()
    # fins
    dorsal = cpath([(-20, -40), (10, -74), (40, -58), (38, -36)], 52, 0.7)
    piece(c, dorsal, "#ffb347", 52, lift=0.8)
    body = cpath(ellipse(0, 0, 72, 46, 60), 53, 0.8)
    piece(c, body, "orange", 53, lift=1.6, grad=0.14)
    c.save()
    c.clipPath(body, doAntiAlias=True)
    c.drawPath(path_of(ellipse(-4, 30, 64, 26, 40)), paper_paint("#ffc36b", 54))
    sc = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.0, Color=col("#e0720f", 0.55))
    for row, yy in enumerate((-18, 2, 20)):
        for k in range(4):
            xx = -6 + k * 18 + (row % 2) * 9
            c.drawArc(skia.Rect.MakeXYWH(xx - 9, yy - 9, 18, 18), -60, 120, False, sc)
    c.restore()
    fin = cpath([(0, 8), (26, 26), (8, 34), (-8, 20)], 55, 0.6)
    c.save()
    c.translate(2, 6)
    c.rotate(math.sin(t * 7) * 15)
    piece(c, fin, "#ffb347", 55, lift=0.7)
    brad(c, 2, 12, 4.5)
    c.restore()
    brad(c, 58, 0, 6)
    eye(c, -38, -10, 14, (-0.5, 0), blink, "happy", 56)
    cheeks(c, -34, 12, 0.1, 8, 0.6)
    mouth(c, -60, 10, 16, mouth_open)
    c.restore()


# ----------------------------------------------------------------------------------------------------
# a duckling (faces left). anchor: waterline under the body
# ----------------------------------------------------------------------------------------------------
def duckling(c, x, y, s=1.0, rot=0.0, t=0.0, mouth_open=0.0, blink=0.0, flap=0.0, nod=0.0, seed=0, look=(-0.5, 0)):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s, s)
    sd = 300 + seed * 10
    body = cpath(ellipse(8, -26, 50, 32, 50), sd, 0.8)
    tail = cpath([(46, -34), (76, -58), (66, -26)], sd + 1, 0.6)
    piece(c, tail, "yellow", sd + 1, lift=1.0)
    piece(c, body, "yellow", sd, lift=1.4, grad=0.14)
    # wing
    c.save()
    c.translate(18, -30)
    c.rotate(-40 * flap * (0.5 + 0.5 * math.sin(t * 30)))
    wing = cpath([(-18, 0), (4, -16), (34, -10), (40, 6), (10, 12)], sd + 2, 0.6)
    piece(c, wing, "#ffe27a", sd + 2, lift=0.8)
    c.restore()
    # head
    c.save()
    c.translate(-26, -62)
    c.rotate(-14 * nod)
    head = cpath(ellipse(0, 0, 27, 26, 40), sd + 3, 0.7)
    piece(c, head, "yellow", sd + 3, lift=1.3, grad=0.14)
    for k, (dx, a) in enumerate(((-4, -20), (2, 0), (8, 20))):
        c.save()
        c.translate(dx * 0.8, -22)
        c.rotate(a)
        piece(c, path_of([(-3, 0), (0, -14), (3, 0)]), "#ffcf2e", sd + 4 + k, lift=0.5, rim=0, shadow=0.15)
        c.restore()
    # beak: two halves
    ob = 12 * clamp(mouth_open)
    upper = cpath([(-20, 2), (-44, 4), (-20, 10)], sd + 7, 0.4)
    lower = cpath([(-20, 10), (-40, 10 + ob * 0.6), (-19, 16)], sd + 8, 0.4)
    c.save()
    c.translate(0, ob * 0.2)
    piece(c, lower, "#f7931e", sd + 8, lift=0.6, rim=0.2)
    c.restore()
    piece(c, upper, "#ff9f1c", sd + 7, lift=0.8, rim=0.2)
    eye(c, -10, -4, 7.5, look, blink, "happy", sd + 9, lift=0.3)
    cheeks(c, -4, 10, 0.1, 6, 0.6)
    c.restore()
    c.restore()


# ----------------------------------------------------------------------------------------------------
# Freddy the frog (faces left). anchor: bottom centre (sitting on a pad)
# ----------------------------------------------------------------------------------------------------
def freddy(c, x, y, s=1.0, rot=0.0, t=0.0, mouth_open=0.0, blink=0.0, jump=0.0, pouch=0.0, tongue=None,
           look=(-0.4, 0.0)):
    """Sitting frog facing left; jump 0..1 kicks the back legs out; tongue: (length, angle_deg) or None."""
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s, s)
    sd = 400
    # far back leg (darker, behind the body)
    for far in (True, False):
        k = 0.85 if far else 1.0
        colr = "#3f8f33" if far else "#4fa63d"
        c.save()
        c.translate(30 + (8 if far else 0), -34)
        c.rotate(-8 + 55 * jump)
        thigh = cpath(ellipse(10, 4, 36 * k, 22 * k, 36), sd + 1 + far, 0.6)
        c.save()
        c.translate(34 * k, 14 * k)
        c.rotate(10 - 75 * jump)
        shin = cpath([(0, -8), (-6, 10), (-56, 18), (-76, 14), (-84, 20), (-60, 26), (-4, 18), (8, 4)],
                     sd + 3 + far, 0.6)
        piece(c, shin, colr, sd + 3 + far, lift=0.9)
        brad(c, 0, 2, 4.5)
        c.restore()
        piece(c, thigh, colr, sd + 1 + far, lift=1.1)
        brad(c, 0, 0, 5)
        c.restore()
        if far:
            body = cpath([(-64, -14), (-58, -62), (-30, -92), (10, -96), (48, -76), (70, -40), (62, -8), (20, 2),
                          (-40, 2)], sd + 10, 0.9, soft=True)
            piece(c, body, "green", sd + 10, lift=1.6, grad=0.14)
            c.save()
            c.clipPath(body, doAntiAlias=True)
            c.drawPath(path_of(ellipse(-26, -6, 44, 34, 40)), paper_paint("#b7e39a", sd + 11))
            for dx, dy, r in ((24, -74, 7), (44, -52, 5), (4, -86, 4), (56, -30, 4)):
                c.drawCircle(dx, dy, r, paper_paint("#3f8f33", sd + 12))
            c.restore()
    # throat pouch
    if pouch > 0.02:
        pp = path_of(ellipse(-44, -34, 16 + 18 * pouch, 12 + 15 * pouch, 36))
        piece(c, pp, "#e4f5b0", sd + 13, lift=0.8, grad=0.2)
    # front arm
    arm = cpath([(-30, -40), (-18, -40), (-34, -2), (-24, 2), (-52, 4), (-50, -2), (-42, -4)], sd + 20, 0.5)
    piece(c, arm, "#4fa63d", sd + 20, lift=0.8)
    # eyes on top
    for dx, sdk in ((-50, 30), (-14, 31)):
        bump = cpath(ellipse(dx, -94, 22, 20, 36), sd + sdk, 0.6)
        piece(c, bump, "green", sd + sdk, lift=1.2)
        eye(c, dx, -96, 15, look, blink, "happy", sd + sdk + 2, lift=0.3)
    if mouth_open > 0.08:
        mouth(c, -36, -58, 44, mouth_open)
    else:
        stroke(c, [(-66, -62), (-46, -52), (-16, -54), (0, -62)], 3.2, "ink")
    cheeks(c, -36, -62, 30, 8, 0.55)
    if tongue is not None:
        ln, ang = tongue
        if ln > 1:
            c.save()
            c.translate(-46, -56)
            c.rotate(ang)
            tp = path_of([(0, -5), (ln, -4), (ln + 6, 0), (ln, 4), (0, 5)])
            piece(c, tp, "#f06a8f", sd + 40, lift=1.0, rim=0.2)
            c.restore()
    c.restore()


def freddy_tongue_tip(x, y, s, ln, ang):
    a = math.radians(ang)
    return x + s * (-46 + math.cos(a) * ln), y + s * (-56 + math.sin(a) * ln)


# ----------------------------------------------------------------------------------------------------
# the whale (faces left). anchor: centre of the body at the waterline
# ----------------------------------------------------------------------------------------------------
def whale(c, x, y, s=1.0, rot=0.0, t=0.0, mouth_open=0.0, blink=0.0, tail=0.0):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s, s)
    sd = 500
    # tail flukes on a split pin
    c.save()
    c.translate(300, -60)
    c.rotate(-28 + 12 * math.sin(t * 2.2) + 20 * tail)
    fl = cpath([(-10, 14), (40, -10), (70, -40), (78, -96), (104, -120), (112, -86), (100, -52), (150, -60),
                (178, -40), (140, -18), (80, 4), (20, 30)], sd + 1, 1.0, soft=True)
    piece(c, fl, "blue", sd + 1, lift=1.4, grad=0.12)
    c.restore()
    body = [(-330, -40), (-300, -150), (-180, -215), (0, -225), (160, -180), (290, -90), (330, -40), (260, 20),
            (0, 40), (-240, 30)]
    bp = cpath(body, sd + 2, 1.2, soft=True)
    piece(c, bp, "blue", sd + 2, lift=2.0, grad=0.16)
    c.save()
    c.clipPath(bp, doAntiAlias=True)
    belly = smooth_path([(-340, -30), (-200, -70), (0, -60), (200, -40), (340, 0), (300, 60), (-340, 60)])
    c.drawPath(belly, paper_paint("#cfe6ff", sd + 3))
    gp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.4, Color=col("#8fb6e6", 0.8))
    for k in range(4):
        yy = -42 + k * 14
        path = skia.Path()
        path.moveTo(-300, yy + 6)
        path.quadTo(-100, yy - 16, 140, yy + 4)
        c.drawPath(path, gp)
    c.restore()
    fin = cpath([(-60, -40), (20, -10), (60, 30), (0, 10), (-50, -10)], sd + 4, 0.8)
    c.save()
    c.translate(-40, -20)
    c.rotate(math.sin(t * 2.5) * 10)
    piece(c, fin, "#2a6cc0", sd + 4, lift=1.2)
    c.restore()
    brad(c, 300, -60, 8)
    brad(c, -40, -20, 7)
    eye(c, -205, -115, 17, (-0.4, 0.1), blink, "happy", sd + 5)
    cheeks(c, -190, -80, 0.1, 16, 0.5)
    if mouth_open > 0.08:
        mouth(c, -270, -62, 70, mouth_open)
    else:
        stroke(c, [(-328, -52), (-290, -40), (-230, -44)], 4, "ink")
    c.drawCircle(-80, -212, 7, skia.Paint(AntiAlias=True, Color=col("#1d4f8f")))
    c.restore()


def whale_blowhole(x, y, s):
    return x + s * -80, y + s * -214


# ----------------------------------------------------------------------------------------------------
# Olive the octopus (faces front). anchor: waterline under the head
# ----------------------------------------------------------------------------------------------------
def _arm(c, x0, y0, ang0, length, width, t, phase, seed, curl=0.0, color="purple", wig=1.0):
    n = 16
    x, y, a = x0, y0, math.radians(ang0)
    spine = [(x, y)]
    for i in range(1, n + 1):
        u = i / n
        a += wig * math.sin(t * 3.4 + phase + u * 3.5) * 0.07 + curl * 0.06 * u * u
        x += math.cos(a) * length / n
        y += math.sin(a) * length / n
        spine.append((x, y))
    pts_l, pts_r = [], []
    for i, (px, py) in enumerate(spine):
        j = min(i, n - 1)
        dx, dy = spine[j + 1][0] - spine[j][0], spine[j + 1][1] - spine[j][1]
        d = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / d, dx / d
        w = width * (1 - 0.78 * i / n) / 2
        pts_l.append((px + nx * w, py + ny * w))
        pts_r.append((px - nx * w, py - ny * w))
    path = smooth_path(pts_l + [spine[-1]] + pts_r[::-1])
    piece(c, path, color, seed, lift=1.0, grad=0.1)
    sp = skia.Paint(AntiAlias=True, Color=col("#e3d0fb"))
    for i in range(2, n - 1, 2):
        px, py = pts_r[i]
        qx, qy = spine[i]
        c.drawCircle(lerp(qx, px, 0.45), lerp(qy, py, 0.45), max(1.6, width * (1 - 0.78 * i / n) * 0.16), sp)
    return spine[-1]


ARM_ANGLES = [172, -164, -140, -118, -62, -40, -16, 8]


def olive(c, x, y, s=1.0, rot=0.0, t=0.0, mouth_open=0.0, blink=0.0, wave=1.0, reach=None, look=(-0.3, 0.1),
          arms=1.0):
    """Arms wave above the water around her head. reach: local (x, y) that arm 1 stretches to, or None.
    Returns the reaching arm's tip in local units."""
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s, s)
    sd = 600
    tip = None
    for i, a in enumerate(ARM_ANGLES):
        side = -1 if i < 4 else 1
        bx = side * (58 - 10 * abs(i - 3.5 + side * 0.0) * 0) + (i - 3.5) * 12
        by = -30 if i in (0, 7) else -40
        ang = a + wave * 22 * math.sin(t * 3.2 + i * 0.9)
        ln = (118 + 18 * math.sin(i * 2.1)) * arms
        if reach is not None and i == 1:
            rx, ry = reach
            ang = math.degrees(math.atan2(ry - by, rx - bx))
            ln = max(20.0, math.hypot(rx - bx, ry - by))
            tip = _arm(c, bx, by, ang, ln, 30, t, i, sd + i, curl=0.0, wig=0.3)
            continue
        if ln > 4:
            _arm(c, bx, by, ang, ln, 30, t, i * 0.8, sd + i, curl=-1.2 * side)
    head = path_of(cut([(88 * math.cos(a), -60 + (98 if math.sin(a) < 0 else 50) * math.sin(a))
                        for a in np.linspace(0, TAU, 64, endpoint=False)], sd + 20, 1.0))
    piece(c, head, "purple", sd + 20, lift=1.8, grad=0.16)
    c.drawCircle(-44, -118, 9, paper_paint("#a57be3", sd + 21))
    c.drawCircle(46, -128, 6, paper_paint("#a57be3", sd + 22))
    for side in (-1, 1):
        bw = cpath([(38, -150), (38 + side * 30, -170), (38 + side * 32, -134)], sd + 23 + side, 0.5)
        piece(c, bw, "pink", sd + 23 + side, lift=1.0)
    c.drawCircle(38, -151, 7.5, paper_paint("rose", sd + 26))
    eye(c, -30, -70, 19, look, blink, "happy", sd + 27)
    eye(c, 30, -70, 19, look, blink, "happy", sd + 28)
    for ex in (-30, 30):
        for k in (-1, 0, 1):
            stroke(c, [(ex + k * 9, -89), (ex + k * 13, -98)], 2.5, "ink")
    cheeks(c, 0, -40, 46, 11, 0.6)
    mouth(c, 0, -34, 26, mouth_open)
    c.restore()
    return tip
