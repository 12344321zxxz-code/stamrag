"""Train roof + tunnel, the build, chorus 1 (lightning surf), manga panels."""
import math

import numpy as np
import skia

from . import cast as K
from . import fxk as FX
from . import hero as Z
from . import lyrics as L
from . import world as Wd
from .core import (BM, H, TAU, W, C, P, clamp, ease_back, ease_in, ease_io, ease_out, font, fract, hash1, inv_lerp,
                   lerp, lin_grad, mix, path_from, poly, pulse, rad_grad, rrect, scale_c, smooth, vnoise)
from .timeline import (A, bar_t, bubble, cam, fly_text, mega_text, melody_ribbon, post, punch, recent_notes, shake,
                       slam_text, zdraw, note_age)

TRAIN_ROOF = 600
CAR_SCROLL = -60
TRAIN_V = 1500.0


def to_screen(pose, zx, zy, size, gx, gy, mirror=False):
    J = Z.joints(pose)
    lx, ly = Z.guitar_point(pose, J, gx, gy)
    return (zx + (-lx if mirror else lx) * size, zy + ly * size)


def note_fx(c, x, pose, zx, zy, size, wire_y=None, color="#7afcff", big=False, mirror=False):
    """Sparks from the guitar on every lead note, arcs up to the overhead wire."""
    notes = recent_notes(x.t, 0.4, min_pitch=50)
    head = to_screen(pose, zx, zy, size, 0.66, 0.0, mirror)
    body = to_screen(pose, zx, zy, size, 0.0, 0.0, mirror)
    for n in notes:
        age = x.t - n[0]
        seed = int(n[0] * 100)
        FX.sparks(c, body[0], body[1], age, seed=seed, n=14 if big else 9, color="#fff39a", speed=520, life=0.4)
        if wire_y is not None and age < 0.12:
            FX.lightning(c, head, (head[0] + (hash1(seed) - 0.5) * 300, wire_y), seed=seed, width=3.5, color=color,
                         a=1 - age / 0.12, branches=1)
        if big and age < 0.15:
            tgt = (head[0] + 500 + hash1(seed) * 500, head[1] - 300 + hash1(seed + 1) * 400)
            FX.lightning(c, head, tgt, seed=seed, width=5, color=color, a=1 - age / 0.15, branches=2)


def train_play_pose(x, yaw=0.35, land_t=None):
    na = note_age(x.t, 50)
    p = Z.play_pose(x.b, x.t, yaw=yaw, fret=Z.pitch_to_fret(A.pitch_at(x.t)), note_age=na, intensity=1.0)
    p["hair_v"] = (-0.12, 0.0)
    p["scarf_wind"] = (-1.0, 0.1)
    p["speed"] = 0.9
    return p


# ----------------------------------------------------------------------------
# TRAIN (bars 9-16)
# ----------------------------------------------------------------------------
def s_train(c, x):
    t = x.t
    ts = TRAIN_V * (t - bar_t(9))
    tunnel = x.bar >= 13
    zoom = punch(x, 0.025) * 1.16
    sx_, sy_ = shake(x, 3 + 5 * x.kick)
    cam(c, zoom, cx=840, cy=500, ox=sx_, oy=sy_)
    if not tunnel:
        Wd.sky(c, [(0, "#08061c"), (0.6, "#24195a"), (1, "#5a2a6a")], -100, 760)
        Wd.stars(c, t, 0.9, seed=13, n=400, h=600)
        Wd.moon(c, 1560, 170, 75)
        Wd.city(c, "far", ts * 0.06 + 300, 700, t, scale=0.9)
        Wd.city(c, "mid", ts * 0.22 + 100, 790, t, scale=1.0)
        Wd.catenary(c, ts, 330, t)
    else:
        tl = t - bar_t(13)
        strobe = x.hat
        lines = [L.lines()[6], L.lines()[7]]
        words = None
        for ln in lines:
            a, wd = ln.state(t, fade_in=0.2, fade_out=0.3)
            if a > 0:
                words = wd
        Wd.tunnel(c, ts, t, strobe, words=words)
    lights = 1.0 if not tunnel else 0.55 + 0.45 * x.hat
    led = None
    for li in (4, 5):
        a, wd = L.lines()[li].state(t, fade_in=0.1, fade_out=0.2)
        if a > 0:
            led = wd
    Wd.train(c, CAR_SCROLL, TRAIN_ROOF, t, led_words=led, led_car=0, lights=lights, track=ts)
    # blob on the next car, taunting
    hop = abs(math.sin(math.pi * x.u))
    br = 72
    bx = 1480 + 40 * math.sin(t * 1.1)
    by = TRAIN_ROOF - br * 0.9 - 110 * hop
    K.draw_blob(c, bx, by, br, t, look=(-1, 0), mood="smug", arms_up=1.0, sy=1 + 0.1 * hop, sx=1 - 0.06 * hop,
                glow=0.6 if tunnel else 0.0)
    K.draw_trophy(c, bx, by - br * 1.55, 110, t, shine=1.0)
    # Zip lands on the roof at bar 9, then plays
    land = x.t - bar_t(9)
    zx, zy, zs = 560, TRAIN_ROOF - 4, 380
    pose = train_play_pose(x)
    if land < 0.3:
        f = clamp(land / 0.3)
        zy = lerp(-300, TRAIN_ROOF - 4, ease_in(f, 2))
        pose = Z.run_pose(x.b, yaw=0.9, t=t)
        pose["feet"] = {1: (0.12, -0.12), -1: (-0.15, -0.08)}
        pose["expr"] = "angry"
    elif land < 0.55:
        pose["root"] = (0.0, -Z.PELVIS_H + 0.09 * (1 - (land - 0.3) / 0.25))
    if tunnel:
        rim_a = 0.4 + 0.6 * x.hat
        zdraw(c, pose, zx, zy, zs, t, rim="#fff0b0", rim_a=rim_a, tint="#2a2050", tint_k=0.35)
    else:
        zdraw(c, pose, zx, zy, zs, t, rim="#8a7ae8", rim_a=0.5)
    note_fx(c, x, pose, zx, zy, zs, wire_y=None if tunnel else 330)
    if 0.25 < land < 0.9:
        FX.sfx(c, "THOOM!", zx, zy - 60, 110, land - 0.28, color="#ffd21f", angle=-6, life=0.6)
    # tunnel mouth sweeping past at bar 13, exit flash at bar 17
    ent = t - bar_t(13)
    if -0.35 < ent < 0.05:
        f = clamp((ent + 0.35) / 0.35)
        xw = lerp(W + 100, -200, f)
        c.drawRect(skia.Rect.MakeLTRB(xw, -200, W + 300, H + 200), P("#050308"))
        c.drawRect(skia.Rect.MakeLTRB(xw - 80, -200, xw, H + 200), P("#2a2238"))
    c.restore()
    FX.hlines(c, t, n=22 if not tunnel else 34, color="#ffffff" if not tunnel else "#fff0b0", a=0.3, y0=100, y1=1000,
              speed=3200, seed=4)
    first = t - bar_t(9)
    ex = t - (bar_t(17) - 0.25)
    return post(bloom=0.4 if not tunnel else 0.55, threshold=0.5, chroma=2.5 * x.snare,
                flash=max(0.6 * math.exp(-first * 6), 0.5 * math.exp(-max(0, t - bar_t(13)) * 8) * (t > bar_t(13)),
                          ease_in(clamp(ex / 0.25), 2) if ex > 0 else 0.0),
                smear=(220 * math.exp(-first * 5), 0), vignette=0.45 if not tunnel else 0.7)


# ----------------------------------------------------------------------------
# BUILD (bars 17-20) and the stop
# ----------------------------------------------------------------------------
def _bridge_scene(c, x, zx, zs, end_x):
    t = x.t
    ts = TRAIN_V * (t - bar_t(17))
    Wd.sky(c, [(0, "#060418"), (0.5, "#1f1a5a"), (0.85, "#5a2a7a"), (1, "#8a3a7a")], -100, 700)
    Wd.stars(c, t, 1.0, seed=17, n=600, h=650)
    Wd.moon(c, 1350, 260, 150, glow_a=0.6)
    Wd.city(c, "far", ts * 0.05 + 1200, 690, t, scale=0.8)
    # river
    c.drawRect(skia.Rect.MakeLTRB(-100, 690, W + 100, H + 100), P(shader=lin_grad((0, 690), (0, H), [(0, "#2a2a6a", 1), (1, "#07081c", 1)])))
    for i in range(60):
        yy = 700 + (i / 60) ** 1.4 * 380
        xx = 1350 + math.sin(i * 7.1 + t * 2) * (20 + i * 3)
        c.drawLine(xx - 40, yy, xx + 40, yy, P("#fff4d0", 0.5, stroke=2))
    # bridge deck + pillars, ending at end_x
    deck_y = TRAIN_ROOF + 350
    c.drawRect(skia.Rect.MakeLTRB(-100, deck_y, end_x, deck_y + 40), P("#141024"))
    for i in range(12):
        px = (i * 420 - ts) % (12 * 420) - 400
        if px < end_x - 40:
            c.drawRect(skia.Rect.MakeLTRB(px, deck_y + 40, px + 50, H + 100), P("#0e0b1c"))
    # broken edge
    c.drawPath(poly([(end_x, deck_y), (end_x + 40, deck_y + 10), (end_x + 10, deck_y + 30), (end_x + 30, deck_y + 42),
                     (end_x - 20, deck_y + 40)]), P("#141024"))


def s_build(c, x):
    t = x.t
    b = x.b
    idx, sb = x.shot([17, 19, 19.5, 20, 20.25, 20.5, 20.75])
    stop_t = A.beat_time(20 * 4 + 3) + 0.06
    if idx == 0:
        ts = TRAIN_V * (t - bar_t(17))
        end_x = lerp(W + 2600, W - 250, clamp((t - bar_t(17)) / (bar_t(19) - bar_t(17))))
        cam(c, punch(x, 0.02))
        _bridge_scene(c, x, 560, 380, end_x)
        Wd.train(c, CAR_SCROLL, TRAIN_ROOF, t, lights=1.0, track=ts)
        # blob leaps off the end and inflates like a balloon
        f = clamp((b - (17 * 4 + 5)) / 3.0)
        bx = lerp(1480, 1900, f)
        by = TRAIN_ROOF - 70 - math.sin(f * math.pi * 0.9) * 360 - f * 120
        br = 72 * (1 + 0.9 * ease_io(f))
        K.draw_blob(c, bx, by, br, t, look=(-1, 0.3), mood="smug" if f < 0.5 else "happy", arms_up=1.0,
                    sx=1 + 0.1 * f, sy=1 + 0.1 * f)
        K.draw_trophy(c, bx, by - br * 1.55, 110 + 40 * f, t, shine=1.0)
        pose = train_play_pose(x)
        zdraw(c, pose, 560, TRAIN_ROOF - 4, 380, t, rim="#b08aff", rim_a=0.5)
        note_fx(c, x, pose, 560, TRAIN_ROOF - 4, 380)
        c.restore()
        slam_text(c, L.lines()[8], t, W / 2, 190, size=104, pal="cool", maxw=1600, kick=x.kick)
        return post(bloom=0.45, threshold=0.5, flash=0.8 * math.exp(-(t - bar_t(17)) * 5), chroma=2 * x.snare)
    # tension cuts every half bar / beat
    cut_age = t - A.beat_time(math.floor(b * (1 if b >= 80 else 0.5)) / (1 if b >= 80 else 0.5))
    bg_cols = ["#1a0f3a", "#3a0f2a", "#0f2a3a", "#2a0f0f", "#101030", "#000000"]
    c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P(bg_cols[min(idx, 5)]))
    FX.radial_lines(c, W / 2, H / 2, t, n=120, inner=0.25, color="#ffffff", a=0.55, seed=idx)
    if idx == 1:      # extreme close-up: eyes
        p = Z.base_pose(0.25)
        p["expr"] = "determined"
        p["eye_glow"] = 0.6
        p["guitar"] = "none"
        p["hair_v"] = (-0.2, -0.05)
        zdraw(c, p, W / 2 + 60, H / 2 + 4200 * 0.83, 4200, t)
        c.drawRect(skia.Rect.MakeLTRB(0, 0, W, 230), P("#000000"))
        c.drawRect(skia.Rect.MakeLTRB(0, H - 230, W, H), P("#000000"))
        FX.sfx(c, "!", 1600, 420, 260, cut_age, color="#ffd21f", life=1.2)
    elif idx == 2:    # gripping the neck
        c.save()
        c.translate(W / 2 - 900, H / 2 + 80)
        c.rotate(-12)
        c.scale(2300, 2300)
        Z.draw_guitar(Z.Ink(c, 0.004), glow=0.8, t=t, string_vib=1.0)
        c.restore()
        p = Z.play_pose(b, t, yaw=0.3, fret=0.6, note_age=0.05)
        FX.sfx(c, "GRIP", 1450, 300, 160, cut_age, color="#7afcff", angle=-8, life=1.2)
    elif idx >= 3:
        n = ["3", "2", "1"][min(idx - 3, 2)] if idx < 6 else ""
        if idx == 3:     # the blob sweating
            K.draw_blob(c, W / 2 + 300, H / 2 + 120, 360, t, look=(-1, 0.2), mood="panic", arms_up=1.0)
            K.draw_trophy(c, W / 2 + 300, H / 2 + 120 - 360 * 1.55, 300, t, shine=1.0)
        elif idx == 4:   # rails rushing to the broken end
            vp = (W / 2, H * 0.42)
            c.drawRect(skia.Rect.MakeLTRB(0, vp[1], W, H), P("#0e0b1c"))
            for side in (-1, 1):
                c.drawLine(vp[0] + side * 30, vp[1], vp[0] + side * 700, H, P("#c9c9d9", 1, stroke=14))
            for j in range(12):
                z = (j + 1 - fract(t * 4))
                yy = vp[1] + 500 / z * 0.6
                if yy < H:
                    c.drawLine(vp[0] - 40 - (yy - vp[1]) * 1.2, yy, vp[0] + 40 + (yy - vp[1]) * 1.2, yy, P("#3a2e5a", 1, stroke=6))
            c.drawRect(skia.Rect.MakeLTRB(0, 0, W, vp[1]), P("#1a1a5a"))
            Wd.moon(c, vp[0], vp[1] - 60, 120)
        elif idx == 5:   # crouch before the jump
            p = Z.base_pose(0.85)
            p["guitar"] = "back"
            p["root"] = (0.0, -Z.PELVIS_H + 0.14)
            p["feet"] = {1: (0.16, 0.0), -1: (-0.2, 0.0)}
            p["hands"] = {1: (-0.3, -0.45), -1: (-0.25, -0.35)}
            p["hand_shape"] = {1: "fist", -1: "fist"}
            p["expr"] = "determined"
            p["lean"] = 30
            zdraw(c, p, W / 2, H - 60, 900, t)
        if n:
            k = ease_back(clamp(cut_age / 0.12), 3)
            c.save()
            c.translate(W - 330 if idx != 4 else W / 2, H / 2 + 90)
            c.scale(k, k)
            FX.outlined_text(c, n, 0, 150, 520, "RubikMonoOne.ttf", fill="#ffd21f", stroke="#120a18", sw=50,
                             extra_stroke="#ff3d8a", glow="#ff3d8a")
            c.restore()
    if idx == 6 or t >= stop_t:
        # THE STOP: freeze on Zip mid-leap against the moon
        c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P(shader=rad_grad((W / 2, H / 2), W * 0.7, [(0, "#2a2a7a", 1), (1, "#05040f", 1)])))
        Wd.moon(c, W / 2 + 120, H / 2 - 60, 330, glow_a=0.9)
        p = Z.base_pose(0.8)
        p["guitar"] = "raise"
        p["gtr_ang"] = -70
        p["feet"] = {1: (0.2, -0.22), -1: (-0.22, -0.04)}
        p["foot_ang"] = {1: -30, -1: 40}
        p["hands"] = {1: (0.1, -1.02), -1: (0.02, -0.98)}
        p["hand_shape"] = {1: "fist", -1: "fist"}
        p["expr"] = "rock"
        p["hair_v"] = (-0.1, -0.25)
        zdraw(c, p, W / 2 + 60, H / 2 + 330, 720, t, sil="#05040f")
        age = t - stop_t
        if age > 0:
            FX.lightning(c, (W / 2 + 60, 60), (W / 2 + 20, H / 2 - 360), seed=int(t * 15), width=6, a=0.9)
            FX.sfx(c, "!!", W / 2 - 520, 280, 300, age, color="#ffffff", life=2.0, angle=-12)
        FX.halftone(c, skia.Rect.MakeLTRB(0, 0, W, H), "#000000", 0.25, cell=10)
        return post(bloom=0.5, threshold=0.45, impact=1.0 if 0 <= age < 0.1 else 0.0, glitch=0.0,
                    flash=0.8 * math.exp(-max(age, 0) * 8) if age > 0 else 0.0, letterbox=0.6)
    return post(bloom=0.4, threshold=0.5, flash=0.55 * math.exp(-cut_age * 7), chroma=4, letterbox=0.4)


# ----------------------------------------------------------------------------
# CHORUS 1: lightning surf (bars 21-27)
# ----------------------------------------------------------------------------
def surf_pose(x, yaw=0.9, arms="out"):
    b = x.b
    p = Z.base_pose(yaw)
    p["guitar"] = "ride"
    p["root"] = (0.0, -Z.PELVIS_H + 0.1 + 0.018 * math.sin(b * math.pi))
    p["feet"] = {1: (0.17, 0.0), -1: (-0.17, 0.0)}
    p["foot_ang"] = {1: 0.0, -1: 0.0}
    p["lean"] = 14
    p["tilt"] = 5 * math.sin(b * math.pi * 0.5)
    if arms == "out":
        p["hands"] = {1: (0.36, -0.66 + 0.04 * math.sin(b * math.pi)), -1: (-0.38, -0.74)}
        p["hand_shape"] = {1: "open", -1: "horns"}
        p["elbow_hint"] = {1: (0.0, 1.0), -1: (0.0, 1.0)}
    p["expr"] = "rock" if (x.bi % 4) in (0, 2) else "grin"
    p["hair_v"] = (-0.3, 0.02)
    p["speed"] = 1.3
    p["scarf_wind"] = (-1.0, 0.0)
    p["gtr_glow"] = 0.8 + 0.2 * x.kick
    p["aura"] = 0.5
    return p


def lightning_trail(c, x0, y0, t, length=1600, seed=0, a=1.0, width=10):
    k = int(t * 20)
    FX.lightning(c, (x0, y0), (x0 - length, y0 + 60 * math.sin(t * 2)), seed=seed + k, width=width, color="#7afcff",
                 a=a, branches=3, jag=0.1)


def s_chorus1(c, x):
    t = x.t
    idx, sb = x.shot([21, 23, 25, 27])
    first = t - bar_t(21)
    strike = math.exp(-first * 3.5)
    if idx in (0, 3):
        scroll = 1800 * (t - bar_t(21))
        rot = 3 * math.sin(x.b * math.pi / 4)
        sx_, sy_ = shake(x, 6 + 10 * x.kick + 30 * strike)
        cam(c, punch(x, 0.03) * (1.0 if idx == 0 else 1.15), rot=rot, ox=sx_, oy=sy_)
        Wd.sky(c, [(0, "#05031a"), (0.5, "#1a1450"), (1, "#40207a")], -200, H)
        Wd.stars(c, t, 1.0, seed=21, n=700, h=H)
        Wd.moon(c, 1500, 220, 120)
        for k in range(5):
            Wd.cloud(c, (k * 520 - scroll * 0.3) % (W + 800) - 400, 300 + (k % 3) * 120, 500, 80, "#b8a8ff", 0.18, seed=k)
        Wd.city(c, "far", scroll * 0.2, H + 180, t, scale=1.0)
        Wd.city(c, "mid", scroll * 0.45, H + 330, t, scale=1.1)
        # blob ahead riding a pink goo comet
        bx = 1450 + 60 * math.sin(t * 1.3)
        by = 430 + 50 * math.sin(t * 2.1)
        for k in range(12):
            gx = bx - 60 - k * 55
            c.drawCircle(gx, by + math.sin(t * 6 + k) * 10, 40 - k * 3, P("#ff5fa8", 0.8 - k * 0.06))
        K.draw_blob(c, bx, by, 90, t, look=(-1, 0), mood="panic" if idx == 3 else "smug", arms_up=1.0, sx=1.15, sy=0.9)
        K.draw_trophy(c, bx, by - 90 * 1.5, 120, t, shine=1.0)
        zx, zy, zs = 620, 700, 420
        lightning_trail(c, zx - 60, zy + 10, t, seed=21)
        pose = surf_pose(x)
        zdraw(c, pose, zx, zy, zs, t, rim="#7afcff", rim_a=0.6)
        if first < 0.4:
            FX.lightning(c, (zx + 40, -100), (zx, zy - 300), seed=int(t * 30), width=16, color="#b8f8ff", a=1 - first / 0.4)
        if idx == 3:
            # final blast at the end of bar 27
            bl = t - (bar_t(28) - A.period * 1.0)
            if bl > 0:
                FX.lightning(c, (zx + 200, zy - 250), (bx, by), seed=int(t * 30), width=14, a=clamp(1 - bl / 0.6))
                FX.sfx(c, "ZZZAP!!", bx - 150, by - 250, 180, bl, color="#fff39a", angle=-10, life=1.0)
        c.restore()
        FX.hlines(c, t, n=30, color="#ffffff", a=0.35, speed=3600, seed=6)
        if idx == 0:
            mega_text(c, L.lines()[9], t, W / 2, 330, size=230, pal="hot", kick=x.kick)
        else:
            slam_text(c, L.lines()[12], t, W / 2, 180, size=110, pal="hot", kick=x.kick)
        return post(bloom=0.55, threshold=0.45, flash=max(0.95 * strike if idx == 0 else 0, 0.5 * x.snare * 0.4),
                    chroma=3 + 4 * x.snare, impact=1.0 if first < 0.07 else 0.0, zoom_blur=0.04 * x.kick)
    if idx == 1:
        # front 3/4 hero shot: surfing at the camera with streaking city lights
        cut = t - bar_t(23)
        cam(c, punch(x, 0.035), rot=-4 + 3 * math.sin(x.b * 0.8))
        c.drawRect(skia.Rect.MakeLTRB(-200, -200, W + 200, H + 200), P(shader=rad_grad((W / 2, H / 2), W, [(0, "#3a1a7a", 1), (1, "#07041a", 1)])))
        rng = np.random.default_rng(3)
        for i in range(90):
            ang = rng.uniform(0, TAU)
            sp = rng.uniform(0.6, 1.4)
            d = ((t * sp * 0.9 + rng.uniform(0, 1)) % 1.0)
            r0 = 80 + d ** 2 * 1400
            r1 = r0 + 40 + d * 300
            cc = ["#ffd27a", "#ff7ac0", "#7afcff"][i % 3]
            c.drawLine(W / 2 + math.cos(ang) * r0, H / 2 + math.sin(ang) * r0, W / 2 + math.cos(ang) * r1,
                       H / 2 + math.sin(ang) * r1, P(cc, 0.8 * d, stroke=2 + 6 * d))
        pose = surf_pose(x, yaw=0.3)
        pose["hands"] = {1: (0.4, -0.9), -1: (-0.4, -0.78)}
        pose["hand_shape"] = {1: "horns", -1: "open"}
        pose["expr"] = "rock"
        zdraw(c, pose, W / 2 + 80, 1080, 900, t, rim="#7afcff", rim_a=0.6)
        c.restore()
        FX.radial_lines(c, W / 2, H / 2, t, n=90, inner=0.4, color="#ffffff", a=0.4)
        slam_text(c, L.lines()[10], t, W / 2, 190, size=108, pal="hot", maxw=1700, kick=x.kick)
        return post(bloom=0.55, threshold=0.45, flash=0.7 * math.exp(-cut * 6), chroma=4 + 4 * x.snare, zoom_blur=0.05)
    # idx == 2: aerial view -- the lightning trail traces the melody across the city grid
    cut = t - bar_t(25)
    cam(c, 1.0 + 0.1 * (cut / 4.3), rot=-8)
    c.drawRect(skia.Rect.MakeLTRB(-300, -300, W + 300, H + 300), P("#0a0718"))
    off = t * 700
    for i in range(-2, 16):
        yy = (i * 160 + off * 0.3) % (18 * 160) - 320
        c.drawLine(-300, yy, W + 300, yy, P("#ffb35a", 0.35, stroke=10))
    for j in range(-2, 22):
        xx = (j * 170 - off) % (24 * 170) - 340
        c.drawLine(xx, -300, xx, H + 300, P("#ffd27a", 0.3, stroke=8))
        for i in range(-1, 12):
            yy = (i * 160 + off * 0.3) % (18 * 160) - 320
            h_ = hash1(j * 31 + i)
            c.drawRect(skia.Rect.MakeLTRB(xx + 20, yy + 20, xx + 150, yy + 140), P(mix("#1a1438", "#2a2058", h_)))
            if h_ > 0.6:
                c.drawRect(skia.Rect.MakeLTRB(xx + 50, yy + 50, xx + 90, yy + 90), P("#7afcff", 0.6))
    def y_of(p):
        return lerp(820, 260, clamp((p - 50) / 30))
    melody_ribbon(c, t, 1150, 520, y_of, back=3.5, color="#7afcff", width=10)
    head = (1150, y_of(A.pitch_at(t)))
    pose = surf_pose(x, yaw=0.95)
    zdraw(c, pose, head[0], head[1] - 10, 230, t, rim="#7afcff", rim_a=0.6)
    bx, by = head[0] + 420 + 40 * math.sin(t * 2), head[1] - 120 + 60 * math.sin(t * 1.7)
    K.draw_blob(c, bx, by, 60, t, look=(-1, 0.3), mood="panic", arms_up=1.0)
    K.draw_trophy(c, bx, by - 90, 80, t, shine=1.0)
    c.restore()
    slam_text(c, L.lines()[11], t, W / 2, 150, size=112, pal="cool", kick=x.kick)
    return post(bloom=0.55, threshold=0.45, flash=0.6 * math.exp(-cut * 6), chroma=3 * x.snare)


# ----------------------------------------------------------------------------
# MANGA PANELS (bars 28-31)
# ----------------------------------------------------------------------------
PAPER = "#f6f1e6"
INK_ = "#111018"


def panel(c, pts, draw_fn, a=1.0, tone=True):
    path = poly(pts)
    c.save()
    c.clipPath(path, skia.ClipOp.kIntersect, True)
    draw_fn()
    if tone:
        FX.halftone(c, path, "#000000", 0.16, cell=9)
    c.restore()
    c.drawPath(path, P(INK_, a, stroke=12))


def mini_blobs(t, b, n=12, seed=4):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x0 = rng.uniform(160, W - 160)
        hop = abs(math.sin(math.pi * (b + i * 0.37)))
        out.append((x0 + math.sin(t * 1.3 + i) * 40, 900 - 60 - 180 * hop, rng.uniform(40, 70), i))
    return out


def s_panels(c, x):
    t = x.t
    b = x.b
    c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P(PAPER))
    bar = x.bar
    if bar < 29:
        age0 = t - bar_t(28)
        def p1():
            c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P("#ffffff"))
            FX.radial_lines(c, 480, 420, t, n=80, inner=0.15, color=INK_, a=0.9, seed=1)
            p = surf_pose(x, yaw=0.5)
            p["guitar"] = "play"
            p["hands"] = {1: None, -1: None}
            zdraw(c, p, 430, 920, 700, t)
            FX.lightning(c, (620, 380), (1000, 200), seed=int(t * 20), width=8, color="#7afcff")
        def p2():
            c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P("#ffe1ef"))
            FX.radial_lines(c, 1180, 420, t, n=80, inner=0.12, color=INK_, a=0.9, seed=2)
            K.draw_blob(c, 1180, 420, 190, t, mood="shock", look=(0, 0), wob=2.0, sx=1.3, sy=0.8)
        def p3():
            c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P("#ffffff"))
            for (mx, my, mr, i) in mini_blobs(t, b, 7, seed=9):
                K.draw_blob(c, 1500 + (mx - W / 2) * 0.25, 800 + (my - 900) * 0.4, mr * 0.9, t + i, mood="panic",
                            look=(math.sin(i), 0))
        a1, a2, a3 = [clamp((b - (28 * 4 + k)) / 0.2) for k in range(3)]
        if a1 > 0:
            panel(c, [(30, 30), (900, 30), (760, 1050), (30, 1050)], p1)
            FX.sfx(c, "ZAP!", 620, 170, 150, t - A.beat_time(112), color="#7afcff", stroke=INK_, life=3)
        if a2 > 0:
            panel(c, [(930, 30), (1890, 30), (1890, 600), (820, 600)], p2)
            FX.sfx(c, "SPLAT!", 1500, 140, 150, t - A.beat_time(113), color="#ff5fa8", stroke=INK_, life=3, angle=8)
        if a3 > 0:
            panel(c, [(810, 630), (1890, 630), (1890, 1050), (760, 1050)], p3)
            FX.sfx(c, "SPLIT?!", 1150, 760, 120, t - A.beat_time(114), color="#ffd21f", stroke=INK_, life=3, angle=-6)
        return post(bloom=0.15, threshold=0.7, flash=0.5 * math.exp(-age0 * 8), grain=0.08, vignette=0.2)
    if bar < 30:
        age0 = t - bar_t(29)
        def full():
            c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P("#ffffff"))
            c.drawRect(skia.Rect.MakeLTRB(0, 900, W, H), P("#d8d0e8"))
            for (mx, my, mr, i) in mini_blobs(t, b):
                K.draw_blob(c, mx, my, mr, t + i, mood="smug" if i % 2 else "happy", look=(math.sin(t + i), 0))
                eighth = A.beat_time(math.floor(b * 2) / 2)
                if (i + int(b * 2)) % 4 == 0:
                    bubble(c, "cringe!", mx + 40, my - mr - 60, (mx, my - mr), t - eighth, 1.0, size=34,
                           fnt="Bangers.ttf", maxw=300)
        panel(c, [(30, 30), (1890, 30), (1890, 1050), (30, 1050)], full)
        p = Z.base_pose(0.3)
        p["guitar"] = "back"
        p["expr"] = "shock"
        c.save()
        c.clipRect(skia.Rect.MakeLTRB(30, 30, 1890, 1050))
        zdraw(c, p, 300, 1350, 1000, t)
        c.restore()
        bubble(c, "THERE'S MORE OF THEM?!", 640, 170, (380, 420), t - bar_t(29) - 0.2, 1.0, size=70, fnt="Bangers.ttf",
               shout=True, maxw=700)
        return post(bloom=0.15, threshold=0.7, flash=0.4 * math.exp(-age0 * 8), grain=0.08, vignette=0.2)
    if bar < 31:
        age0 = t - bar_t(30)
        k = min(3, int(b - 30 * 4))
        boxes = [(30, 30, 945, 530), (975, 30, 1890, 530), (30, 560, 945, 1050), (975, 560, 1890, 1050)]
        for q in range(k + 1):
            x0, y0, x1, y1 = boxes[q]
            qa = t - A.beat_time(30 * 4 + q)
            def draw_q(q=q, x0=x0, y0=y0, x1=x1, y1=y1, qa=qa):
                c.drawRect(skia.Rect.MakeLTRB(x0, y0, x1, y1), P(["#fff4d0", "#e0f8ff", "#ffe1ef", "#f0e8ff"][q]))
                FX.radial_lines(c, (x0 + x1) / 2, (y0 + y1) / 2, t, n=50, inner=0.1, color=INK_, a=0.5, seed=q + 5)
                p = Z.base_pose(0.6)
                p["guitar"] = "raise"
                swing = ease_out(clamp(qa / 0.2))
                p["gtr_ang"] = lerp(-120, 10, swing)
                p["hands"] = {1: (0.12, -0.8 + 0.3 * swing), -1: (0.05, -0.78 + 0.3 * swing)}
                p["expr"] = "angry"
                zx_ = x0 + 230 if q % 2 == 0 else x1 - 230
                zdraw(c, p, zx_, y1 + 20, 430, t, mirror=(q % 2 == 1))
                fly = ease_out(clamp(qa / 0.5))
                mb = (x0 + 560 + fly * 260) if q % 2 == 0 else (x1 - 560 - fly * 260)
                K.draw_blob(c, mb, y0 + 300 - fly * 180, 70, t + q, mood="dizzy", look=(0, 0))
            panel(c, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], draw_q)
            FX.sfx(c, "BONK!", x0 + 620, y0 + 120, 120, qa - 0.12, color="#ffd21f", stroke=INK_, life=2, angle=-10 + q * 6)
        return post(bloom=0.15, threshold=0.7, flash=0.3 * math.exp(-(t - A.beat_time(math.floor(b))) * 10), grain=0.08,
                    vignette=0.2)
    # bar 31: the minis merge into a giant blob; the stop hits at 92.3
    lt = t - bar_t(31)
    stop = A.beat_time(31 * 4 + 3) + 0.2
    grow = ease_io(clamp(lt / 1.4))
    def big():
        c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P(shader=lin_grad((0, 0), (0, H), [(0, "#1a1040", 1), (1, "#40205a", 1)])))
        Wd.city(c, "mid", 400, H + 60, t, scale=1.0)
        for (mx, my, mr, i) in mini_blobs(t, b):
            f = grow
            K.draw_blob(c, lerp(mx, W / 2, f), lerp(my, 760, f), mr * (1 - f), t + i, mood="panic")
        if grow > 0.2:
            K.draw_blob(c, W / 2, 1000 - 460 * grow, 470 * grow, t, mood="smug", look=(-0.5, 0.3), ground=1000,
                        glow=0.4)
    panel(c, [(30, 30), (1890, 30), (1890, 1050), (30, 1050)], big, tone=False)
    p = Z.base_pose(0.4)
    p["guitar"] = "back"
    p["expr"] = "shock" if lt > 1.0 else "determined"
    zdraw(c, p, 320, 1320, 820, t)
    if lt > 1.1:
        bubble(c, "...UH OH.", 520, 250, (360, 480), lt - 1.1, 1.0, size=76, fnt="Bangers.ttf", maxw=500)
    age = t - stop
    return post(bloom=0.35, threshold=0.55, impact=1.0 if 0 <= age < 0.1 else 0.0, flash=0.6 * math.exp(-lt * 6),
                grain=0.08, letterbox=0.3 * clamp(age / 0.2) if age > 0 else 0.0)
