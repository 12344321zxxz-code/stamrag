"""The nine scenes. Each s_<name>(c, t) draws one frame at film time t (seconds); static scenery comes
from baked Layers, the cast and anything that moves is drawn live on top."""
import math

import skia

from . import anim as N
from . import cast as A
from . import world as Wd
from .anim import TL, blink, m, mouth
from .craft import (RAINBOW, TAU, Cam, Layer, clamp, col, cpath, ease_back, ease_io, ease_out, ellipse, grow,
                    hash1, lerp, letter, path_of, piece, smooth, smoother, thread, torn_piece, word_layout)
from .core import H, W
from .timeline import title_letter_times

_LAYERS = {}


def layers(name, build):
    if name not in _LAYERS:
        if len(_LAYERS) >= 3:
            _LAYERS.pop(next(iter(_LAYERS)))
        _LAYERS[name] = build()
    return _LAYERS[name]


def sc(name):
    return TL.scene(name)


def lt(name, t):
    return t - sc(name)["t0"]


def ease_in_out_pos(a, b, t0, t1, t):
    return lerp(a, b, smoother(clamp((t - t0) / (t1 - t0))))


def cloud_layer(x, y, w, h, color, seed, lift=2.0, blur=0.0):
    return Layer((x - w * 0.75, y - h * 1.3, x + w * 0.75, y + h * 0.9),
                 lambda c: Wd.cloud(c, x, y, w, h, color, seed, lift), blur=blur)


def strip_layer(y, amp, period, color, seed, x0=-400, x1=W + 400, lift=1.2, scallop=False, shine=True, depth=None):
    b = H + 80 if depth is None else y + depth
    return Layer((x0, y - amp * 1.6 - 30, x1, b),
                 lambda c: Wd.water_strip(c, x0, x1, y, amp, period, color, seed, lift=lift, scallop=scallop,
                                          shine=shine, depth=(b - y + 40)))


def slide(t, k, amp=18.0, speed=0.35):
    return amp * math.sin(t * speed * TAU * 0.5 + k * 1.3), 4.0 * math.sin(t * 1.1 + k * 2.1)


def pop_in(t, t0, d=0.4, k=2.0):
    return ease_back(clamp((t - t0) / d), k)


def push(c, t, focus, amount, t_in, t_out, d=1.3):
    """Gentle camera push toward focus between t_in and t_out (canvas already saved by the caller)."""
    z = 1 + amount * smoother(clamp((t - t_in) / d)) * (1 - smoother(clamp((t - t_out) / d)))
    if z > 1.0005:
        c.translate(focus[0], focus[1])
        c.scale(z, z)
        c.translate(-focus[0], -focus[1])


# ====================================================================================================
# TITLE
# ====================================================================================================
TITLE_LINES = [("Pip", 250, 300), ("and the", 92, 452), ("Paper Rainbow", 158, 588)]


def _build_title():
    L = {"sky": Layer((0, 0, W, H), lambda c: Wd.sky(c, -10, -10, W + 10, H + 10, "sky_blue", "sky_blue2", 3))}
    L["clouds"] = [cloud_layer(260, 170, 300, 120, "cloud", 11), cloud_layer(1180, 120, 260, 100, "cloud", 12),
                   cloud_layer(1460, 330, 220, 86, "cloud", 13, 1.6), cloud_layer(520, 420, 200, 80, "cloud", 14, 1.4)]
    L["sea"] = [strip_layer(820, 12, 260, "sea4", 21), strip_layer(880, 14, 300, "sea", 22),
                strip_layer(955, 12, 240, "sea2", 23)]
    return L


def _title_glyphs():
    out = []
    for text, size, y in TITLE_LINES:
        lay, width = word_layout(text, size)
        for ch, xo, adv in lay:
            if ch != " ":
                out.append((ch, W / 2 - width / 2 + xo, y, size))
    return out


def title_letters(c, t):
    times = title_letter_times(TL.d)
    glyphs = _title_glyphs()
    k_col = 0
    for k, ((ch, x, y, size), tk) in enumerate(zip(glyphs, times)):
        if 3 <= k < 9:
            colr = "#6b5a50"
        else:
            colr = RAINBOW[k_col % 6]
            k_col += 1
        u = clamp((t - (tk - 0.3)) / 0.3)
        if u <= 0:
            continue
        yy = lerp(y - 460, y, ease_out(u, 2))
        land = t - tk
        sq = 1.0 - 0.14 * math.exp(-max(0.0, land) * 12) * (land > 0)
        rot = (hash1(k * 3 + 1) - 0.5) * 12
        bob = 4 * math.sin(t * 2.2 + k * 0.7) if land > 0 else 0
        letter(c, ch, x, yy + bob, size, colr, seed=k * 5 + 1, rot=rot + (1 - u) * 25 * (hash1(k) - 0.5),
               scale=1.0 * (1 + (1 - sq) * 0.2) if land > 0 else 1.0, lift=2.2)


def s_title(c, t):
    L = layers("title", _build_title)
    L["sky"].draw(c)
    Wd.sun(c, 1700, 190, 88, t, seed=5)
    for i, cl in enumerate(L["clouds"]):
        cl.draw(c, dx=24 * math.sin(t * 0.25 + i * 1.7))
    title_letters(c, t)
    for k in range(2):
        dx, dy = slide(t, k)
        L["sea"][k].draw(c, dx, dy)
    hop = 34 * math.sin(math.pi * clamp((t - TL.lines[0]["t0"]) / 0.45))
    N.pip_draw(c, t, 960, 880, 0.86, mood="happy", look=(0.0, -0.8), hop=hop)
    dx, dy = slide(t, 2)
    L["sea"][2].draw(c, dx, dy)


# ====================================================================================================
# GARDEN
# ====================================================================================================
PUDDLE = (870, 842, 340, 74)
DROPS = [(150, 330), (390, 250), (610, 400), (860, 290), (1080, 380), (1290, 260), (1500, 360), (1720, 280),
         (1880, 410)]


def _build_garden():
    WW = W + 900
    L = {"sky": Layer((0, 0, W, H), lambda c: Wd.sky(c, -10, -10, W + 10, H + 10, "sky_grey", "sky_grey2", 4))}
    L["clouds"] = [cloud_layer(250, 120, 420, 150, "cloud_grey", 31, 2.4), cloud_layer(820, 80, 360, 130, "cloud_grey2", 32, 2.2),
                   cloud_layer(1360, 130, 440, 160, "cloud_grey", 33, 2.4), cloud_layer(1900, 90, 380, 140, "cloud_grey2", 34, 2.2),
                   cloud_layer(2500, 120, 420, 150, "cloud_grey", 35, 2.4)]

    def far(c):
        Wd.hills(c, -200, WW, 610, 34, 820, "hill3", 41, lift=1.2)
        Wd.hills(c, -200, WW, 660, 26, 560, "hill", 42, lift=1.4)
    L["far"] = Layer((-200, 480, WW, 900), far, blur=1.2)

    def mid(c):
        Wd.tree(c, 2380, 740, 1.0, 52, canopy="#6aa84f")
        Wd.fence(c, -150, WW, 735, 130, 43)
        Wd.bush(c, 700, 735, 330, 120, "#5d9c49", 44)
        Wd.bush(c, 1500, 740, 280, 110, "#6aa84f", 45)
        Wd.bush(c, 2050, 738, 320, 120, "#5d9c49", 46)
        Wd.tree(c, 250, 760, 1.05, 51, canopy="#5f9e4f")
    L["mid"] = Layer((-150, 280, WW, 800), mid)

    def ground(c):
        Wd.hills(c, -200, WW, 745, 8, 420, "grass", 61, lift=1.6, amp_t=2.4)
        for k, (x, colr) in enumerate([(470, "pink"), (560, "lilac"), (650, "#ffffff"), (1330, "pink"), (1420, "#ffffff"),
                                       (1510, "lilac"), (1960, "pink"), (2060, "#ffffff")]):
            Wd.flower(c, x, 760 - 30 * (k % 2), 22, colr, 70 + k, center="yellow")
        Wd.grass_tuft(c, 360, 790, 60, "grass2", 81)
        Wd.grass_tuft(c, 1560, 800, 70, "grass2", 82)
        # puddle + the little stream that leaves it to the right
        x, y, rx, ry = PUDDLE
        pts = ellipse(x, y, rx, ry, 120)
        stream = [(x + rx * 0.7, y - 40), (x + rx + 200, y - 30), (x + rx + 520, y - 40), (WW, y - 46), (WW, y + 30),
                  (x + rx + 520, y + 38), (x + rx + 200, y + 40), (x + rx * 0.7, y + 40)]
        from .craft import union, from_path, spline
        u = union([path_of(pts), path_of(spline(stream, 8))])
        outline = from_path(u, 4)
        torn_piece(c, outline, "#dfe9e4", 91, lift=0.4, rim_w=5.0, amp=2.5)
        piece(c, path_of(grow(outline, -8)), "puddle", 92, lift=0.0, shadow=0, grad=0.0, rim=0)
        for k in range(5):
            gx = x - 200 + k * 95
            piece(c, cpath([(gx, y - 20 + k * 6), (gx + 60, y - 24 + k * 6), (gx + 56, y - 18 + k * 6),
                            (gx - 4, y - 14 + k * 6)], 93 + k, 0.5), "#b5cfdc", 93 + k, lift=0.2, shadow=0, rim=0,
                  alpha=0.9)
    L["ground"] = Layer((-200, 560, WW, H + 60), ground)

    def fg(c):
        Wd.grass_tuft(c, 80, 1110, 260, "grass2", 101, blades=9, lift=2.4)
        Wd.grass_tuft(c, 1850, 1120, 300, "#4f8f35", 102, blades=9, lift=2.4)
        Wd.flower(c, 1680, 990, 58, "rose", 103, stem=False, lift=2.6)
    L["fg"] = Layer((-150, 780, W + 250, H + 80), fg, blur=5.0)
    return L


def garden_pip(t):
    """Pip's pose in the garden (world coords)."""
    x, y = 860.0, 842.0
    ts = m("garden.sail")
    if t > ts:
        u = clamp((t - ts) / (sc("garden")["t1"] - ts + 0.5))
        x = lerp(860, 2200, smooth(u) ** 1.2)
        y = lerp(842, 812, smooth(u))
    return x, y


def s_garden(c, t):
    c.save()
    push(c, t, (870, 760), 0.16, m("garden.hello") + 0.4, m("garden.tracker") - 0.6)
    _garden(c, t)
    c.restore()
    N.draw_tracker(c, t)


def _garden(c, t):
    L = layers("garden", _build_garden)
    lt_ = t - sc("garden")["t0"]
    px, py = garden_pip(t)
    ts = m("garden.sail")
    cam_x = W / 2 + max(0.0, px - 860) * 0.92
    cam = Cam(cam_x + 30 * math.sin(lt_ * 0.2), H / 2, 1.0 + 0.05 * smooth(clamp(lt_ / 12)))
    L["sky"].draw(c)
    # hanging raindrops, pulled up one by one when the rain stops
    c.save()
    cam.apply(c, 0.45)
    tr = m("garden.rain_up")
    for k, (dx, dy) in enumerate(DROPS):
        pull = ease_io(clamp((t - tr - 0.25 - k * 0.09) / 0.8))
        y = lerp(dy, -120, pull)
        sway = 7 * math.sin(t * 1.5 + k)
        x = dx + sway
        thread(c, dx, 60, x, y - 28, sag=3)
        Wd.raindrop(c, x, y, 1.0, 700 + k)
    c.restore()
    c.save()
    cam.apply(c, 0.3)
    for i, cl in enumerate(L["clouds"]):
        cl.draw(c, dx=30 * math.sin(t * 0.15 + i))
    c.restore()
    c.save()
    cam.apply(c, 0.55)
    L["far"].draw(c)
    c.restore()
    c.save()
    cam.apply(c, 0.8)
    L["mid"].draw(c)
    c.restore()
    c.save()
    cam.apply(c, 1.0)
    L["ground"].draw(c)
    # two falling drops at 'Drip, drop!'
    td = m("garden.drips")
    for k, (fx, off) in enumerate(((820, 0.0), (950, 0.42))):
        u = (t - td - off + 0.3) / 0.3
        if 0 <= u < 1:
            Wd.raindrop(c, fx, lerp(-60, PUDDLE[1] - 20, u ** 2), 0.9, 720 + k)
        r = t - td - off
        if 0 <= r < 1.2:
            a = 1 - r / 1.2
            c.drawOval(skia.Rect.MakeXYWH(fx - 20 - 90 * r, PUDDLE[1] - 20 - 6 * r - 10 * r, 40 + 180 * r, 18 + 34 * r),
                       skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3,
                                  Color=col("#e6f2f7", 0.8 * a)))
    # Pip
    th, ti = m("garden.hello"), m("garden.idea")
    mood, look = "happy", None
    if th + 0.7 <= t < ti:
        mood, look = "sad", (0.1, -0.9)
    elif ti <= t < ti + 2.2:
        mood, look = "excited", (0.0, -0.3)
    hop = 40 * math.sin(math.pi * clamp((t - ti) / 0.5)) + 26 * math.sin(math.pi * clamp((t - th) / 0.4))
    rot = -6 * smooth(clamp((t - ts) / 0.5)) if t > ts else 0
    pose = N.pip_draw(c, t, px, py, 0.9, rot, mood, look, hop)
    # idea bulb
    if ti <= t < ti + 2.0:
        u = pop_in(t, ti, 0.35)
        fade = clamp((ti + 2.0 - t) / 0.3)
        bx, by = A.pip_point(*pose, 70, -330)
        c.save()
        c.translate(bx, by)
        c.scale(u, u)
        piece(c, cpath(ellipse(0, 0, 34, 38, 40), 740, 0.6), "#ffe36b", 740, lift=2.0, alpha=fade)
        piece(c, cpath([(-16, 30), (16, 30), (14, 52), (-14, 52)], 741, 0.5), "#b8b8b8", 741, lift=1.0, alpha=fade)
        for k in range(7):
            a = math.radians(-180 + k * 30)
            N.star(c, 58 * math.cos(a), 58 * math.sin(a) - 4, 9, fade, rot=t * 90)
        c.restore()
    # puddle lip in front of the hull
    x, y, rx, ry = PUDDLE
    if t < ts + 0.8:
        c.save()
        c.clipPath(path_of(ellipse(x, y, rx - 6, ry - 6, 90)), doAntiAlias=True)
        lip = Wd.wave_band(px - 260, px + 260, py + 34, 4, 140, 95, 200)
        piece(c, path_of(lip), "puddle", 96, lift=0.2, shadow=0.15, grad=0, rim=0.5)
        c.restore()
    c.restore()
    c.save()
    cam.apply(c, 1.25)
    L["fg"].draw(c, dx=0)
    c.restore()


# ====================================================================================================
# shared pieces for the water scenes
# ====================================================================================================
PIP_Y = 820
FRONT_Y = PIP_Y + 44


def grey_sky_layers(seed):
    L = {"sky": Layer((0, 0, W, H), lambda c: Wd.sky(c, -10, -10, W + 10, H + 10, "sky_grey", "sky_grey2", seed))}
    L["clouds"] = [cloud_layer(300, 110, 440, 150, "cloud_grey", seed + 1, 2.4),
                   cloud_layer(960, 70, 380, 130, "cloud_grey2", seed + 2, 2.2),
                   cloud_layer(1620, 120, 460, 160, "cloud_grey", seed + 3, 2.4)]
    return L


def draw_sky(c, L, t, cam=None):
    L["sky"].draw(c)
    for i, cl in enumerate(L["clouds"]):
        cl.draw(c, dx=26 * math.sin(t * 0.14 + i * 2))


def draw_strips(c, strips, t, k0=0):
    for k, st in enumerate(strips):
        dx, dy = slide(t, k + k0)
        st.draw(c, dx, dy)


def enter_x(t, t0, x1, d=1.4, x0=-260.0):
    return lerp(x0, x1, ease_out(clamp((t - t0) / d), 3))


def pip_at(t, scene, x=700.0):
    """Pip sails in from the left at the start of a scene: -> (x, look, tilt)."""
    t0 = sc(scene)["t0"]
    px = enter_x(t, t0 - 0.2, x)
    tilt = -5 * (1 - clamp((t - t0) / 1.4))
    return px, None, tilt


def exit_drift(t, scene):
    """Pip keeps sailing right at the end of a scene (so the wipe feels like travel)."""
    t1 = sc(scene)["t1"]
    return 220 * smooth(clamp((t - (t1 - 1.1)) / 1.6))


def hop_on(t, t0, h=30, d=0.45):
    return h * math.sin(math.pi * clamp((t - t0) / d))


def word_hop(t, color):
    return hop_on(t, m(f"{color}.word"), 32, 0.45)


# ====================================================================================================
# RED: Lulu the ladybug on a big leaf by the stream
# ====================================================================================================
LEAF_TIP = (1150, 548)
LULU_HOME = (1335, 506)


def _build_red():
    L = grey_sky_layers(110)

    def far(c):
        Wd.hills(c, -200, W + 200, 600, 30, 700, "hill3", 111, lift=1.2)
        for k in range(9):
            Wd.bush(c, -60 + k * 250, 650, 300, 140, ["#7fae6a", "#6f9e5c", "#8aba74"][k % 3], 112 + k, 1.6)
    L["far"] = Layer((-200, 380, W + 200, 800), far, blur=1.8)

    def bank(c):
        Wd.hills(c, -200, W + 200, 672, 10, 380, "grass", 121, lift=1.6, amp_t=2.4)
        for k, x in enumerate((140, 420, 760, 1040)):
            Wd.flower(c, x, 690 - 12 * (k % 2), 18, ["pink", "#ffffff", "lilac", "pink"][k], 122 + k)
        Wd.reeds(c, 1780, 700, 260, 127, n=6)
        Wd.grass_tuft(c, 600, 700, 60, "grass2", 128)
    L["bank"] = Layer((-200, 420, W + 200, 820), bank)
    L["back"] = [strip_layer(700, 7, 320, "water4", 131), strip_layer(760, 9, 280, "water", 132)]

    def stem(c):
        pts = [(2010, 330), (1850, 350), (1700, 410), (1580, 470)]
        from .craft import stroke
        stroke(c, pts, 20, "#4f8a3a")
        stroke(c, pts, 14, "#5f9e47")
        Wd.leaf(c, 1600, 470, 480, 110, 171, "#4f9a3f", 141, lift=2.2, bend=-0.05)
        Wd.leaf(c, 1850, 352, 220, 60, -130, "#5fae4a", 142, lift=2.0)
    L["leaf"] = Layer((1050, 250, W + 150, 640), stem)
    L["front"] = [strip_layer(FRONT_Y, 8, 260, "water2", 151), strip_layer(960, 8, 230, "water3", 152)]

    def fg(c):
        Wd.rock(c, 1700, 1080, 150, 80, "rock", 161, 2.2)
        Wd.rock(c, 1540, 1090, 90, 50, "rock2", 162, 2.0)
        Wd.grass_tuft(c, 120, 1100, 280, "#4f8f35", 163, blades=9, lift=2.4)
    L["fg"] = Layer((-150, 760, W + 150, H + 80), fg, blur=4.5)
    return L


def lulu_pos(t, pose):
    tg, ta = m("red.give"), N.attach_time("red")
    hx, hy = LULU_HOME
    tgt = N.mast_point(pose, 0)
    tx, ty = tgt[0] + 30, tgt[1] - 70
    if t < tg:
        return hx, hy, 0.0, 0.0
    if t < ta:
        u = smoother(clamp((t - tg) / (ta - tg)))
        x, y = N.bez((hx, hy), ((hx + tx) / 2, min(hy, ty) - 200), (tx, ty), u)
        return x, y, 1.0, -8 * math.sin(u * math.pi)
    u = smoother(clamp((t - ta - 0.15) / 1.1))
    x, y = N.bez((tx, ty), ((hx + tx) / 2, min(hy, ty) - 180), (hx, hy), u)
    return x, y, 1.0 if u < 0.98 else 0.0, 8 * math.sin(u * math.pi)


def s_red(c, t):
    L = layers("red", _build_red)
    c.save()
    push(c, t, (1010, 640), 0.1, m("red.reveal") + 0.3, m("red.word") + 0.5)
    draw_sky(c, L, t)
    L["far"].draw(c, dx=10 * math.sin(t * 0.1))
    L["bank"].draw(c)
    draw_strips(c, L["back"], t)
    L["leaf"].draw(c, dy=3 * math.sin(t * 1.2))
    px, look, tilt = pip_at(t, "red")
    px += exit_drift(t, "red")
    tr = m("red.reveal")
    if t > tr:
        look = (0.7, -0.35)
    pose_guess = (px, PIP_Y, 0.95, tilt)
    lx, ly, fly, lrot = lulu_pos(t, pose_guess)
    # Lulu pops up on the leaf, then flies over with the red streamer
    if t > tr - 0.05:
        sc_ = pop_in(t, tr, 0.45) * 0.95
        mo = mouth("lulu", t)
        c.save()
        if fly > 0:
            N.flying_ribbon(c, "red", lx + 10, ly + 40, t, rot=80, s=0.8) if t < N.attach_time("red") else None
        A.lulu(c, lx, ly + 3 * math.sin(t * 1.2), sc_, lrot, t, mo, blink(t, 11), wings=fly,
               look=(-0.6, 0.1) if lx > px else (0.6, 0.2), legs=float(N.speaking("lulu", t)))
        c.restore()
    pose = N.pip_draw(c, t, px, PIP_Y, 0.95, tilt, "happy", look, word_hop(t, "red"))
    draw_strips(c, L["front"], t, 2)
    L["fg"].draw(c)
    ta = N.attach_time("red")
    if ta - 0.1 <= t <= ta + 0.6:
        N.sparkle_trail(c, lambda u: N.mast_point(pose, 0), t, ta - 0.1, ta + 0.2, seed=5)
    c.restore()
    N.draw_tracker(c, t)
    N.color_word(c, t)


# ====================================================================================================
# ORANGE: Finn the goldfish leaps out of the stream
# ====================================================================================================
def _build_orange():
    L = grey_sky_layers(210)

    def far(c):
        Wd.hills(c, -200, W + 200, 590, 36, 640, "hill3", 211, lift=1.2)
        Wd.tree(c, 260, 660, 0.8, 212, canopy="#7fae6a")
        Wd.tree(c, 1560, 650, 0.9, 213, canopy="#6f9e5c")
        for k in range(8):
            Wd.bush(c, 100 + k * 270, 660, 280, 120, ["#8aba74", "#7fae6a"][k % 2], 214 + k, 1.6)
    L["far"] = Layer((-200, 120, W + 200, 800), far, blur=1.8)

    def bank(c):
        Wd.hills(c, -200, W + 200, 676, 9, 420, "grass3", 221, lift=1.6, amp_t=2.4)
        Wd.reeds(c, 240, 700, 240, 222, n=5)
        Wd.reeds(c, 1320, 700, 200, 223, n=4)
        for k, x in enumerate((560, 700, 1650, 1800)):
            Wd.flower(c, x, 694, 17, ["#ffffff", "pink", "lilac", "#ffffff"][k], 224 + k)
    L["bank"] = Layer((-200, 420, W + 200, 820), bank)
    L["back"] = [strip_layer(704, 7, 300, "water4", 231), strip_layer(766, 9, 270, "water", 232)]

    def rocks(c):
        Wd.rock(c, 1640, 812, 170, 90, "rock", 241, 2.0)
        Wd.rock(c, 1790, 822, 100, 60, "rock2", 242, 1.8)
        Wd.grass_tuft(c, 1640, 736, 50, "moss", 243, blades=6, lift=0.8)
    L["rocks"] = Layer((1420, 640, W + 60, 900), rocks)
    L["front"] = [strip_layer(FRONT_Y, 8, 250, "water2", 251), strip_layer(962, 8, 220, "water3", 252)]

    def fg(c):
        Wd.grass_tuft(c, 1800, 1100, 300, "#4f8f35", 261, blades=9, lift=2.4)
        Wd.leaf(c, -60, 1040, 420, 110, -24, "#3d7f31", 262, lift=2.4)
    L["fg"] = Layer((-200, 700, W + 200, H + 80), fg, blur=4.5)
    return L


FINN_TALK = (1170, 836)


def finn_state(t, pose):
    """-> (x, y, rot, visible_above_water)"""
    tr, tg = m("orange.reveal"), m("orange.give")
    if t < tr - 0.02:
        return None
    if t < tr + 0.55:
        u = (t - tr) / 0.55
        x = lerp(1560, 1250, u)
        y = FRONT_Y + 30 - 330 * math.sin(math.pi * u)
        return x, y, lerp(-40, 40, u), 1.0
    if t < tr + 1.1:
        u = smooth((t - tr - 0.55) / 0.55)
        return lerp(1300, FINN_TALK[0], u), lerp(FRONT_Y + 60, FINN_TALK[1], u), 0.0, 1.0
    if t < tg + 0.05:
        return FINN_TALK[0], FINN_TALK[1] + 4 * math.sin(t * 2.4), 6 * math.sin(t * 1.7), 1.0
    if t < tg + 1.0:
        u = (t - tg - 0.05) / 0.95
        x = lerp(FINN_TALK[0], 1420, u)
        y = FRONT_Y + 10 - 360 * math.sin(math.pi * u)
        return x, y, lerp(-50, 60, u) - 360 * smooth(u), 1.0
    if t < tg + 1.6:
        return None
    u = smooth(clamp((t - tg - 1.6) / 0.6))
    return lerp(1480, 1300, u), lerp(FRONT_Y + 60, FINN_TALK[1] + 10, u), 0.0, 1.0


def s_orange(c, t):
    L = layers("orange", _build_orange)
    c.save()
    push(c, t, (960, 700), 0.08, m("orange.reveal") + 0.3, m("orange.word") + 0.5)
    draw_sky(c, L, t)
    L["far"].draw(c, dx=10 * math.sin(t * 0.1))
    L["bank"].draw(c)
    draw_strips(c, L["back"], t)
    L["rocks"].draw(c, dy=2 * math.sin(t))
    px, look, tilt = pip_at(t, "orange")
    px += exit_drift(t, "orange")
    tr, tg = m("orange.reveal"), m("orange.give")
    if t > tr:
        look = (0.7, -0.2)
    pose0 = (px, PIP_Y, 0.95, tilt)
    st = finn_state(t, pose0)
    if st is not None:
        x, y, rot, _ = st
        A.finn(c, x, y, 0.95, rot, t, mouth("finn", t), blink(t, 21), wag=1.0)
    pose = N.pip_draw(c, t, px, PIP_Y, 0.95, tilt, "happy", look, word_hop(t, "orange"))
    draw_strips(c, L["front"], t, 2)
    # splashes
    for ts_, sx in ((tr, 1560), (tr + 0.55, 1250), (tg + 0.05, FINN_TALK[0]), (tg + 1.0, 1420)):
        Wd.splash_drops(c, sx, FRONT_Y + 6, (t - ts_) / 0.8, 10, 110, 150, int(ts_ * 10), "#cfeaf8")
    # the streamer is flicked off his tail at the top of the leap
    ta = N.attach_time("orange")
    N.hand_off(c, "orange", t, (FINN_TALK[0] + 100, FINN_TALK[1] - 300), N.mast_point(pose, 1), tg + 0.5, ta, 120, 7)
    L["fg"].draw(c)
    c.restore()
    N.draw_tracker(c, t)
    N.color_word(c, t)


# ====================================================================================================
# YELLOW: three ducklings on the pond
# ====================================================================================================
def _build_yellow():
    L = grey_sky_layers(310)

    def far(c):
        Wd.hills(c, -200, W + 200, 600, 26, 760, "hill3", 311, lift=1.2)
        Wd.hills(c, -200, W + 200, 640, 22, 520, "hill2", 312, lift=1.3)
    L["far"] = Layer((-200, 420, W + 200, 800), far, blur=2.0)

    def bank(c):
        Wd.willow(c, 1520, 690, 1.0, 321)
        Wd.hills(c, -200, W + 200, 684, 8, 400, "grass", 322, lift=1.6, amp_t=2.4)
        Wd.reeds(c, 140, 708, 300, 323, n=7)
        Wd.reeds(c, 420, 704, 220, 324, n=4, cattails=False)
        Wd.bush(c, 900, 690, 260, 100, "#6aa84f", 325)
    L["bank"] = Layer((-200, 60, W + 200, 830), bank)
    L["back"] = [strip_layer(712, 6, 340, "water4", 331), strip_layer(770, 8, 300, "water", 332)]

    def pads(c):
        Wd.lily_pad(c, 420, 760, 70, 17, 341)
        Wd.lily_pad(c, 560, 774, 50, 12, 342)
        Wd.water_lily(c, 430, 752, 40, 343)
    L["pads"] = Layer((300, 690, 660, 820), pads)
    L["front"] = [strip_layer(FRONT_Y, 7, 280, "water2", 351), strip_layer(962, 8, 240, "water3", 352)]

    def fg(c):
        Wd.reeds(c, 1860, 1120, 520, 361, n=6, lift=2.4)
        Wd.grass_tuft(c, 60, 1110, 260, "#4f8f35", 362, blades=8, lift=2.4)
    L["fg"] = Layer((-200, 480, W + 200, H + 80), fg, blur=4.5)
    return L


def ducks_state(t):
    tr = m("yellow.reveal")
    out = []
    for k in range(3):
        u = ease_out(clamp((t - tr + 0.2 - k * 0.18) / 2.4), 3)
        x = lerp(2150 + 190 * k, 1150 + 175 * k, u)
        out.append(x)
    return out


def s_yellow(c, t):
    L = layers("yellow", _build_yellow)
    c.save()
    push(c, t, (1010, 740), 0.1, m("yellow.reveal") + 0.3, m("yellow.word") + 0.5)
    draw_sky(c, L, t)
    L["far"].draw(c)
    L["bank"].draw(c)
    draw_strips(c, L["back"], t)
    L["pads"].draw(c, dy=2 * math.sin(t * 1.3))
    px, look, tilt = pip_at(t, "yellow")
    px += exit_drift(t, "yellow")
    tr, tg = m("yellow.reveal"), m("yellow.give")
    if t > tr + 0.5:
        look = (0.8, 0.0)
    xs = ducks_state(t)
    for k in (2, 1, 0):
        mem = f"duck{k + 1}"
        mo = mouth(mem, t)
        nod = mo * 0.8 + 0.2 * math.sin(t * 5 + k) * (t < tr + 2.5)
        flap = 1.0 if tg <= t < tg + 0.9 else 0.0
        A.duckling(c, xs[k], FRONT_Y + 2 + 4 * math.sin(t * 2.2 + k * 1.3), 0.95, 3 * math.sin(t * 2 + k), t, mo,
                   blink(t, 30 + k), flap, nod, seed=k)
    pose = N.pip_draw(c, t, px, PIP_Y, 0.95, tilt, "happy", look, word_hop(t, "yellow"))
    draw_strips(c, L["front"], t, 2)
    ta = N.attach_time("yellow")
    N.hand_off(c, "yellow", t, (xs[1] - 20, FRONT_Y - 110), N.mast_point(pose, 2), tg + 0.25, ta, 200, 9)
    L["fg"].draw(c)
    c.restore()
    N.draw_tracker(c, t)
    N.color_word(c, t)


# ====================================================================================================
# GREEN: Freddy the frog and the lily pads
# ====================================================================================================
FAR_PAD = (1640, 792)
NEAR_PAD = (1245, 850)


def _build_green():
    L = grey_sky_layers(410)

    def far(c):
        Wd.hills(c, -200, W + 200, 600, 30, 700, "hill3", 411, lift=1.2)
        for k in range(8):
            Wd.bush(c, -40 + k * 280, 655, 300, 130, ["#6f9e5c", "#7fae6a", "#5f9150"][k % 3], 412 + k, 1.6)
    L["far"] = Layer((-200, 380, W + 200, 800), far, blur=2.0)

    def bank(c):
        Wd.hills(c, -200, W + 200, 680, 9, 380, "grass2", 421, lift=1.6, amp_t=2.4)
        Wd.reeds(c, 90, 706, 320, 422, n=7)
        Wd.reeds(c, 1040, 700, 250, 423, n=5)
        Wd.reeds(c, 1860, 706, 330, 424, n=6)
    L["bank"] = Layer((-200, 300, W + 200, 830), bank)
    L["back"] = [strip_layer(708, 6, 320, "water4", 431), strip_layer(768, 8, 290, "water", 432)]

    def pads(c):
        Wd.lily_pad(c, FAR_PAD[0], FAR_PAD[1] + 6, 110, 24, 441)
        Wd.lily_pad(c, 380, 770, 80, 18, 442)
        Wd.lily_pad(c, 900, 782, 64, 15, 443)
        Wd.water_lily(c, 360, 764, 46, 444)
        Wd.water_lily(c, 1760, 790, 40, 445)
    L["pads"] = Layer((260, 700, W, 850), pads)

    def near_pad(c):
        Wd.lily_pad(c, NEAR_PAD[0], NEAR_PAD[1] + 8, 160, 34, 451, rot=-4)
    L["near"] = Layer((1060, 790, 1440, 910), near_pad)
    L["front"] = [strip_layer(FRONT_Y + 14, 7, 260, "water2", 461), strip_layer(962, 8, 240, "water3", 462)]

    def fg(c):
        Wd.lily_pad(c, 300, 1050, 300, 70, 471, rot=6)
        Wd.reeds(c, 1880, 1120, 560, 472, n=6, lift=2.4)
    L["fg"] = Layer((-100, 460, W + 200, H + 80), fg, blur=4.5)
    return L


def freddy_state(t):
    tr = m("green.reveal")
    if t < tr:
        return FAR_PAD[0], FAR_PAD[1] - 4, 0.9, 0.0, 0.0
    u = clamp((t - tr) / 0.7)
    if u < 1:
        x = lerp(FAR_PAD[0], NEAR_PAD[0], u)
        y = lerp(FAR_PAD[1] - 4, NEAR_PAD[1], u) - 320 * math.sin(math.pi * u)
        return x, y, lerp(0.9, 1.2, u), min(1.0, 1.6 * math.sin(math.pi * u)), -14 * math.sin(math.pi * u)
    land = t - tr - 0.7
    return NEAR_PAD[0], NEAR_PAD[1] + 6 * math.exp(-land * 8) * math.sin(land * 30), 1.2, 0.0, 0.0


def s_green(c, t):
    L = layers("green", _build_green)
    c.save()
    push(c, t, (980, 720), 0.1, m("green.reveal") + 0.3, m("green.word") + 0.5)
    draw_sky(c, L, t)
    L["far"].draw(c)
    L["bank"].draw(c)
    draw_strips(c, L["back"], t)
    L["pads"].draw(c, dy=2 * math.sin(t * 1.1))
    L["near"].draw(c, dy=3 * math.sin(t * 1.4))
    Wd.dragonfly(c, 700 + 380 * math.sin(t * 0.6), 420 + 80 * math.sin(t * 1.3), 0.9, t, 5,
                 rot=10 * math.cos(t * 0.6))
    px, look, tilt = pip_at(t, "green")
    px += exit_drift(t, "green")
    tr, tg = m("green.reveal"), m("green.give")
    if t > tr:
        look = (0.8, -0.1)
    x, y, s_, jump, rot = freddy_state(t)
    pose = (px, PIP_Y, 0.95, tilt)
    tongue = None
    ta = N.attach_time("green")
    if tg <= t < tg + 0.85:
        mx, my = N.mast_point(pose, 3)
        ox, oy = x + s_ * -46, y + s_ * -56
        full = math.hypot(mx - ox, my - oy) / s_
        ang = math.degrees(math.atan2(my - oy, mx - ox))
        ext = ease_out(clamp((t - tg - 0.1) / 0.35)) if t < ta + 0.05 else 1 - smooth(clamp((t - ta - 0.05) / 0.3))
        tongue = (full * ext, ang)
    ribbit = TL.line("green", "freddy")["t0"]
    pouch = math.sin(math.pi * clamp((t - ribbit) / 0.5)) + 0.6 * math.sin(math.pi * clamp((t - tr - 0.9) / 0.5))
    A.freddy(c, x, y + 2 * math.sin(t * 1.4), s_, rot, t, mouth("freddy", t), blink(t, 41), jump, pouch, tongue)
    if tongue is not None and t < ta:
        tx, ty = A.freddy_tongue_tip(x, y, s_, tongue[0], tongue[1])
        N.flying_ribbon(c, "green", tx - 40, ty + 10, t, rot=tongue[1] + 180, s=0.8, idx=3)
    pose = N.pip_draw(c, t, px, PIP_Y, 0.95, tilt, "happy", look, word_hop(t, "green"))
    draw_strips(c, L["front"], t, 2)
    if ta - 0.1 <= t <= ta + 0.6:
        N.sparkle_trail(c, lambda u: N.mast_point(pose, 3), t, ta - 0.1, ta + 0.2, seed=6)
    L["fg"].draw(c)
    c.restore()
    N.draw_tracker(c, t)
    N.color_word(c, t)


# ====================================================================================================
# the sea (BLUE, PURPLE and the FINALE share it)
# ====================================================================================================
def _sea_layers(seed, props=True, sky_=("sky_grey", "sky_grey2")):
    L = {"sky": Layer((0, 0, W, H), lambda c: Wd.sky(c, -10, -10, W + 10, H + 10, sky_[0], sky_[1], seed))}
    L["clouds"] = [cloud_layer(300, 110, 440, 150, "cloud_grey", seed + 1, 2.4),
                   cloud_layer(1000, 70, 380, 130, "cloud_grey2", seed + 2, 2.2),
                   cloud_layer(1650, 130, 460, 160, "cloud_grey", seed + 3, 2.4)]

    def horizon(c):
        piece(c, Wd.wave_band(-300, W + 300, 580, 3, 500, seed + 4, 600), "#9ccbea", seed + 4, lift=0.8, grad=0)
        if props:
            Wd.island(c, 330, 596, 0.8, seed + 5)
            Wd.lighthouse(c, 1690, 606, 0.72, seed + 6)
    L["horizon"] = Layer((-300, 250, W + 300, 900), horizon, blur=1.0)
    L["back"] = [strip_layer(640, 9, 330, "sea4", seed + 10, scallop=True),
                 strip_layer(718, 11, 300, "sea", seed + 11, scallop=True)]
    L["mid"] = [strip_layer(800, 12, 280, "sea2", seed + 12, scallop=True)]
    L["front"] = [strip_layer(FRONT_Y + 8, 12, 260, "sea3", seed + 13, scallop=True),
                  strip_layer(968, 10, 240, "#1f5a96", seed + 14, scallop=True)]
    return L


def draw_gulls(c, t):
    for k in range(3):
        x = (300 + k * 520 + t * (40 + 12 * k)) % (W + 400) - 200
        y = 300 + 60 * k + 20 * math.sin(t * 0.8 + k)
        Wd.gull(c, x, y, 0.9 - 0.15 * k, t, k)


def spout(c, x, y, u, s=1.0, seed=0):
    """Whale spout: a fan of paper droplets, 0 < u < 1."""
    if u <= 0 or u >= 1:
        return
    h = 300 * s * math.sin(math.pi * min(1.0, u * 1.6)) ** 0.7
    for k in range(16):
        a = (k / 15 - 0.5) * 1.3
        r = h * (0.55 + 0.45 * math.cos(a * 1.2))
        dx = math.sin(a) * r * 0.55
        dy = -math.cos(a) * r
        sz = 13 * s * (1 - 0.5 * abs(a))
        c.save()
        c.translate(x + dx, y + dy + 30 * u * u * abs(a))
        c.rotate(math.degrees(a) * 0.6)
        piece(c, cpath([(0, -sz * 1.5), (sz * 0.8, 0), (0, sz), (-sz * 0.8, 0)], seed + k, 0.3, soft=True),
              "#dff1fb" if k % 2 else "#bfe3f7", seed + k, lift=1.0, rim=0.2)
        c.restore()
    col_ = [(x - 16 * s, y), (x - 8 * s, y - h * 0.8), (x + 8 * s, y - h * 0.8), (x + 16 * s, y)]
    piece(c, path_of(col_), "#e9f6fc", seed + 30, lift=0.8, alpha=0.9, rim=0.2)


WHALE_X = 1360


def whale_y(t):
    tr = m("blue.reveal")
    u = clamp((t - tr) / 1.1)
    return lerp(1180, 772, ease_out(u, 3)) + 6 * math.sin(t * 1.1)


def _build_blue():
    return _sea_layers(510)


def s_blue(c, t):
    L = layers("blue", _build_blue)
    c.save()
    push(c, t, (1000, 700), 0.08, m("blue.reveal") + 0.3, m("blue.word") + 0.5)
    draw_sky(c, L, t)
    L["horizon"].draw(c)
    draw_gulls(c, t)
    draw_strips(c, L["back"], t)
    px, look, tilt = pip_at(t, "blue", x=640)
    px += exit_drift(t, "blue")
    tr, tg = m("blue.reveal"), m("blue.give")
    if t > tr + 0.3:
        look = (0.8, -0.5)
    wy = whale_y(t)
    bx, by = A.whale_blowhole(WHALE_X, wy, 0.9)
    su = (t - tg) / 1.4
    spout(c, bx, by + 8, su, 1.0, 520)
    if t > tr:
        A.whale(c, WHALE_X, wy, 0.9, -2 + 2 * math.sin(t * 0.9), t, mouth("whale", t), blink(t, 51))
    draw_strips(c, L["mid"], t, 2)
    mood = "wow" if tr + 0.6 < t < tr + 2.4 and not N.speaking("pip", t) else "happy"
    pose = N.pip_draw(c, t, px, PIP_Y, 0.9, tilt, mood, look, word_hop(t, "blue"))
    draw_strips(c, L["front"], t, 3)
    Wd.splash_drops(c, WHALE_X - 60, FRONT_Y - 10, (t - tr - 0.55) / 1.0, 16, 260, 260, 77, "#cfeaf8", 1.3)
    ta = N.attach_time("blue")
    N.hand_off(c, "blue", t, (bx, by - 290), N.mast_point(pose, 4), tg + 0.55, ta, 120, 11)
    c.restore()
    N.draw_tracker(c, t)
    N.color_word(c, t)


# ====================================================================================================
# PURPLE: Olive the octopus
# ====================================================================================================
OLIVE_X = 1290


def _build_purple():
    L = _sea_layers(610, props=False)

    def rocks(c):
        Wd.island(c, 420, 598, 0.55, 611)
        Wd.rock(c, 1720, 840, 190, 150, "rock", 612, 2.2)
        Wd.rock(c, 1560, 856, 110, 70, "rock2", 613, 2.0)
        for k in range(4):
            Wd.reeds(c, 1650 + k * 40, 760, 120, 614 + k, n=2, cattails=False, color="#3f8f6a")
        # a starfish
        pts = []
        for k in range(10):
            a = math.radians(-90 + 36 * k)
            r = 34 if k % 2 == 0 else 15
            pts.append((1750 + r * math.cos(a), 700 + r * math.sin(a)))
        piece(c, cpath(pts, 615, 0.6), "#ff8a5c", 615, lift=1.2)
    L["rocks"] = Layer((200, 400, W + 60, 1000), rocks)
    return L


def olive_state(t):
    tr = m("purple.reveal")
    u = clamp((t - tr) / 0.9)
    return OLIVE_X, lerp(1150, FRONT_Y + 6, ease_back(u, 1.4)) + 5 * math.sin(t * 1.6), u


def s_purple(c, t):
    L = layers("purple", _build_purple)
    c.save()
    push(c, t, (960, 740), 0.1, m("purple.reveal") + 0.3, m("purple.word") + 0.5)
    draw_sky(c, L, t)
    L["horizon"].draw(c)
    draw_gulls(c, t + 7)
    draw_strips(c, L["back"], t)
    L["rocks"].draw(c)
    px, look, tilt = pip_at(t, "purple", x=660)
    px += exit_drift(t, "purple")
    tr, tg = m("purple.reveal"), m("purple.give")
    if t > tr + 0.3:
        look = (0.8, -0.2)
    ox, oy, arms = olive_state(t)
    pose0 = (px, PIP_Y, 0.95, tilt)
    draw_strips(c, L["mid"], t, 2)
    reach = None
    ta = N.attach_time("purple")
    if t > tr - 0.05:
        s_ = 1.25
        if tg <= t < ta + 0.6:
            mx, my = N.mast_point(pose0, 5)
            lx, ly = (mx - ox) / s_, (my - oy) / s_
            ext = ease_out(clamp((t - tg) / 0.7)) if t < ta else 1 - smooth(clamp((t - ta - 0.1) / 0.5))
            bx, by = -44.5, -40
            reach = (lerp(bx - 60, lx, ext), lerp(by - 60, ly, ext))
        tip = A.olive(c, ox, oy, s_, 2 * math.sin(t * 1.3), t, mouth("olive", t), blink(t, 61),
                      wave=1.0 if t < tg else 0.5, reach=reach, arms=max(0.05, arms))
        if reach is not None and t < ta and tip is not None:
            N.flying_ribbon(c, "purple", ox + tip[0] * s_ - 50, oy + tip[1] * s_ + 6, t, rot=0, s=0.8, idx=5)
    pose = N.pip_draw(c, t, px, PIP_Y, 0.95, tilt, "happy", look, word_hop(t, "purple"))
    draw_strips(c, L["front"], t, 3)
    for k in range(9):
        bu = (t - tr - k * 0.09) / 1.2
        if 0 < bu < 1:
            bxx = OLIVE_X - 80 + 160 * hash1(k + 3)
            c.drawCircle(bxx + 10 * math.sin(bu * 9 + k), FRONT_Y + 20 - 260 * bu, 7 + 6 * hash1(k),
                         skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.5,
                                    Color=col("#eaf6fd", 1 - bu)))
    if ta - 0.1 <= t <= ta + 0.6:
        N.sparkle_trail(c, lambda u: N.mast_point(pose, 5), t, ta - 0.1, ta + 0.2, seed=8)
    c.restore()
    N.draw_tracker(c, t)
    N.color_word(c, t)


# ====================================================================================================
# FINALE: six streamers become a rainbow, the sky clears, everyone cheers
# ====================================================================================================
RB_C = (960, 1070)
RB_R = [826, 760, 694, 628, 562, 496]
RB_W = 64


def _build_finale():
    L = _sea_layers(710, props=True)
    L["blue_sky"] = Layer((0, 0, W, H), lambda c: Wd.sky(c, -10, -10, W + 10, H + 10, "sky_blue", "sky_blue2", 711))
    L["white"] = [cloud_layer(150, 1010, 420, 170, "cloud", 712, 2.4), cloud_layer(1770, 1010, 420, 170, "cloud", 713, 2.4),
                  cloud_layer(420, 230, 260, 100, "cloud", 714, 1.8), cloud_layer(1430, 300, 220, 90, "cloud", 715, 1.6)]
    L["rainbow"] = Layer((0, 150, W, H), lambda c: [Wd.rainbow_band(c, RB_C[0], RB_C[1], RB_R[i], RB_W, RAINBOW[i],
                                                                    720 + i, 1.0, lift=1.8) for i in range(6)])
    return L


def ribbon_flight(t, k, pose):
    """Where streamer k is after the launch: rising from the mast to its spot in the sky."""
    tl_ = m("finale.launch") + k * 0.08
    tb = m(f"finale.beat{k}")
    x0, y0 = N.mast_point(pose, k)
    a = math.asin(clamp((RB_C[1] - 600) / RB_R[k]))
    hover = (RB_C[0] - RB_R[k] * math.cos(a) - 20, 580 - 12 * k)
    if t < tb:
        u = smoother(clamp((t - tl_) / 0.9))
        p1 = ((x0 + hover[0]) / 2 + 200, min(y0, hover[1]) - 380)
        x, y = N.bez((x0, y0), p1, hover, u)
        return x + 10 * math.sin(t * 5 + k) * u, y + 8 * math.cos(t * 4 + k) * u
    return hover


def s_finale(c, t):
    c.save()
    push(c, t, (900, 700), 0.10, sc("finale")["t0"] - 2.0, m("finale.up") + 0.2, d=1.5)
    _finale(c, t)
    c.restore()
    _finale_hud(c, t)


def _finale(c, t):
    L = layers("finale", _build_finale)
    th = m("finale.hooray")
    clear = smooth(clamp((t - th + 0.3) / 1.0))
    L["sky"].draw(c)
    if clear > 0:
        L["blue_sky"].draw(c, alpha=clear)
    Wd.sun(c, 1720, lerp(700, 190, ease_out(clamp((t - th) / 1.4))), 92, t, seed=730,
           mouth_open=0.0) if t > th - 0.1 else None
    for i, cl in enumerate(L["clouds"]):
        away = ease_io(clamp((t - th + 0.2) / 1.2))
        cl.draw(c, dx=26 * math.sin(t * 0.14 + i * 2) + (i - 1 + (0.3 if i == 1 else 0)) * 1400 * away,
                dy=-260 * away)
    for i, cl in enumerate(L["white"][2:]):
        u = pop_in(t, th + 0.3 + 0.2 * i, 0.5)
        if u > 0.01:
            cl.draw(c, dx=20 * math.sin(t * 0.3 + i))
    L["horizon"].draw(c)
    # rainbow arcs, one per beat
    done = t > m("finale.beat5") + 0.6
    if done:
        L["rainbow"].draw(c)
    else:
        for k in range(6):
            tb = m(f"finale.beat{k}")
            if t >= tb:
                sw = smoother(clamp((t - tb) / 0.5))
                Wd.rainbow_band(c, RB_C[0], RB_C[1], RB_R[k], RB_W, RAINBOW[k], 720 + k, sw, lift=1.8)
    # friends: the whale far out, Olive and Freddy on the water
    dx, dy = slide(t, 0)
    L["back"][0].draw(c, dx, dy)
    if t > th - 0.1:
        wx, wy = 380, 690 + 260 * (1 - ease_out(clamp((t - th) / 0.8)))
        spout(c, *A.whale_blowhole(wx, wy, 0.5), ((t - th - 0.7) % 3.0) / 1.4, 0.6, 740)
        A.whale(c, wx, wy + 4 * math.sin(t), 0.5, 0, t, mouth("whale", t), blink(t, 51))
    dx, dy = slide(t, 1)
    L["back"][1].draw(c, dx, dy)
    draw_strips(c, L["mid"], t, 2)
    if t > th - 0.1:
        u = ease_back(clamp((t - th - 0.15) / 0.8), 1.3)
        A.olive(c, 1610, lerp(1150, 848, u), 0.85, 0, t, mouth("olive", t), blink(t, 61), wave=1.2)
        pad_y = lerp(1100, 850, ease_out(clamp((t - th - 0.3) / 0.8)))
        Wd.lily_pad(c, 1300, pad_y + 8, 130, 28, 451)
        A.freddy(c, 1300, pad_y + 2 * math.sin(t * 2), 0.95, 0, t, mouth("freddy", t), blink(t, 41),
                 jump=0.0, pouch=0.5 + 0.5 * math.sin(t * 6) if t < th + 1 else 0.0)
    # Pip
    px, py = 900.0, PIP_Y - 4
    look = None
    mood = "happy"
    tu = m("finale.up")
    hop = hop_on(t, TL.lines_in("finale")[0]["t0"], 26) + sum(hop_on(t, tu + 0.25 + 0.36 * k, 30, 0.34) for k in range(3))
    if t > m("finale.launch"):
        look = (0.0, -1.0)
    if t > th:
        mood = "joy" if not N.speaking("pip", t) else "happy"
        hop += hop_on(t, th, 40, 0.5) + hop_on(t, m("finale.thanks"), 30, 0.45)
    if m("finale.look") <= t < m("finale.look") + 1.4:
        mood, look = "wow" if not N.speaking("pip", t) else "happy", (0.0, -1.0)
    pose = N.pip_draw(c, t, px, py, 1.0, 0.0, mood, look, hop)
    # streamers flying up to become arcs
    for k in range(6):
        tl_ = m("finale.launch") + k * 0.08
        tb = m(f"finale.beat{k}")
        if tl_ <= t < tb + 0.05:
            x, y = ribbon_flight(t, k, pose)
            N.flying_ribbon(c, RAINBOW[k], x, y, t, rot=20 * math.sin(t * 3 + k), s=1.0, idx=k)
            N.sparkle_trail(c, lambda u, k=k: ribbon_flight(tl_ + u * (tb - tl_), k, pose), t, tl_, tb, seed=30 + k)
        if tb <= t < tb + 0.8:
            a = math.radians(180 + 180 * smoother(clamp((t - tb) / 0.5)))
            N.star(c, RB_C[0] + RB_R[k] * math.cos(a), RB_C[1] + RB_R[k] * math.sin(a), 22, 1 - (t - tb) / 0.8,
                   rot=t * 200)
    # ducklings and Finn in front
    if t > th - 0.1:
        for k in range(3):
            u = ease_out(clamp((t - th - 0.2 - 0.12 * k) / 0.9))
            A.duckling(c, lerp(-200, 330 + 165 * k, u), FRONT_Y + 26 + 4 * math.sin(t * 2.2 + k), 0.95,
                       4 * math.sin(t * 3 + k), t, mouth(f"duck{k + 1}", t), blink(t, 30 + k),
                       flap=1.0 if t < th + 1.5 else 0.0, nod=0.3 * math.sin(t * 6 + k), seed=k)
        for j, t0 in enumerate((th + 0.4, th + 2.2, th + 4.0, th + 5.8)):
            u = (t - t0) / 0.8
            if 0 <= u < 1:
                A.finn(c, lerp(1500, 1250, u), FRONT_Y + 40 - 330 * math.sin(math.pi * u), 0.7, lerp(-40, 40, u), t,
                       mouth("finn", t), 0, 1.0)
    draw_strips(c, L["front"], t, 3)
    # Lulu circles Pip
    if t > th - 0.1:
        a = (t - th) * 1.6
        u = clamp((t - th) / 0.6)
        lx = lerp(W + 150, px + 330 * math.cos(a), u)
        ly = lerp(300, 520 + 90 * math.sin(a), u)
        A.lulu(c, lx, ly, 0.7, 10 * math.sin(a), t, mouth("lulu", t), blink(t, 11), wings=1.0,
               look=(-0.6 if math.sin(a) > 0 else 0.6, 0.2))
    Wd.confetti(c, t, seed=5, n=110, t0=th)


def _finale_hud(c, t):
    N.draw_tracker(c, t)
    # THE END
    te = m("finale.end")
    if t > te - 0.05:
        lay, width = word_layout("The End", 150)
        for k, (ch, xo, adv) in enumerate(lay):
            u = clamp((t - te - k * 0.06) / 0.35)
            if u <= 0 or ch == " ":
                continue
            letter(c, ch, W / 2 - width / 2 + xo, 560 + 6 * math.sin(t * 2.5 + k), 150, RAINBOW[k % 6],
                   seed=900 + k, rot=(hash1(k + 77) - 0.5) * 10, scale=ease_back(u, 2.2), lift=2.4)
