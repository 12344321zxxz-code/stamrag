"""Guitar solo showdown, inside the blob, the final chorus at dawn, victory jam, ending."""
import math

import numpy as np
import skia

from mv import props as mvp
from . import cast as K
from . import fxk as FX
from . import hero as Z
from . import lyrics as L
from . import world as Wd
from .core import (BM, H, TAU, W, C, P, clamp, ease_back, ease_in, ease_io, ease_out, font, fract, hash1, inv_lerp,
                   lerp, lin_grad, mix, path_from, poly, pulse, rad_grad, rrect, scale_c, smooth, vnoise)
from .scenes_b import surf_pose, to_screen
from .timeline import (A, bar_t, bubble, cam, float_text, mega_text, melody_ribbon, post, punch, recent_notes, shake,
                       sky_text, slam_text, zdraw, note_age)

SFX_WORDS = ["ZAP!", "KRAK!", "BZZT!", "ZWING!", "SHRED!", "WHAM!", "VRRM!", "FZZT!"]


# ----------------------------------------------------------------------------
# SOLO: kaiju showdown (bars 32-39)
# ----------------------------------------------------------------------------
BLOB_X, BLOB_GROUND, BLOB_R = 1360, 1010, 430


def solo_play_pose(x, yaw=0.45, power=0.0):
    na = note_age(x.t, 55)
    p = Z.play_pose(x.b, x.t, yaw=yaw, fret=Z.pitch_to_fret(A.pitch_at(x.t)), note_age=na, intensity=1.0,
                    strum_rate=4)
    p["lean"] = -8 - 6 * power
    p["aura"] = 0.3 + 0.7 * power
    p["eye_glow"] = power
    p["gtr_glow"] = clamp(p["gtr_glow"] + 0.4 * power)
    p["hair_v"] = (-0.05, -0.12 * power)
    p["expr"] = "rock" if int(x.b * 2) % 3 else "focus"
    return p


def blob_hits(t):
    hits = recent_notes(t, 0.5, min_pitch=60)
    return sum(math.exp(-(t - n[0]) * 9) for n in hits)


def hit_point(seed, cx, cy, r):
    a = -math.pi * (0.15 + 0.7 * hash1(seed))
    return cx + math.cos(a) * r * 0.8, cy + math.sin(a) * r * 0.8


def solo_bolts(c, x, head, bx, by, br, big=1.0):
    for n in recent_notes(x.t, 0.6, min_pitch=60):
        age = x.t - n[0]
        seed = int(n[0] * 1000)
        tgt = hit_point(seed, bx, by, br)
        if age < 0.16:
            col_ = ["#7afcff", "#ffd21f", "#ff7ac0", "#b8ff7a"][seed % 4]
            FX.lightning(c, head, tgt, seed=seed, width=5 * big, color=col_, a=1 - age / 0.16, branches=3)
        FX.sparks(c, tgt[0], tgt[1], age, seed=seed, n=16, color="#fff39a", speed=700, life=0.45, size=4)
        if n[2] >= 72:
            FX.sfx(c, SFX_WORDS[seed % len(SFX_WORDS)], tgt[0] + 60, tgt[1] - 60, 110 * big, age, color="#ffd21f",
                   angle=-12 + (seed % 5) * 6, life=0.6, seed=seed)


def solo_backdrop(c, x, horizon=880):
    t = x.t
    Wd.sky(c, [(0, "#04031a"), (0.5, "#1c1450"), (0.85, "#4a1a6a"), (1, "#8a2a6a")], -300, horizon)
    Wd.stars(c, t, 1.0, seed=33, n=500, h=700)
    Wd.moon(c, 1640, 170, 80)
    # storm flashes on the kick
    c.drawRect(skia.Rect.MakeLTRB(-300, -300, W + 300, horizon), P("#b8a8ff", 0.12 * x.kick))
    Wd.city(c, "far", 700, horizon - 40, t, scale=1.0)


def s_solo(c, x):
    t = x.t
    idx, sb = x.shot([32, 34, 36, 38])
    hits = blob_hits(t)
    power = clamp((x.bar - 36) / 0.5) * (1 - clamp((x.bar - 38) / 1.0) * 0.5)
    first = t - bar_t(32)
    br = BLOB_R * (1 - 0.1 * clamp((x.bar - 32) / 8))
    if idx in (0, 2, 3):
        zoom = {0: 1.0, 2: 1.08, 3: 0.92}[idx]
        rot = 2.5 * math.sin(t * 0.8) if idx == 2 else 0.0
        sx_, sy_ = shake(x, 4 + 10 * x.kick + 14 * hits)
        cam(c, zoom * punch(x, 0.02), rot=rot, ox=sx_, oy=sy_)
        solo_backdrop(c, x)
        Wd.city(c, "mid", 300, 1040, t, scale=1.1)
        # the kaiju blob
        by = BLOB_GROUND - br * 0.9
        lunge = ease_in(clamp((x.bar - 39.4) / 0.6), 2)
        K.draw_blob(c, BLOB_X - 500 * lunge, by - 60 * lunge, br * (1 + 0.3 * lunge), t, look=(-1, -0.2),
                    mood="angry" if hits < 1.2 else "shock", sx=1 + 0.08 * hits, sy=1 - 0.06 * hits,
                    wob=1 + 1.5 * hits, ground=BLOB_GROUND, glow=0.4, lw=8)
        Wd.city(c, "near", 900, H + 260, t, scale=0.85)
        # radio tower + Zip on top
        top = 430
        Wd.radio_tower(c, 420, H + 40, top, t, color="#3a2c6e")
        pose = solo_play_pose(x, power=power)
        zdraw(c, pose, 420, top - 14, 330, t, rim="#7afcff", rim_a=0.6)
        head = to_screen(pose, 420, top - 14, 330, 0.66, 0.0)
        # the melody ribbon streams out of the headstock toward the blob
        def y_of(p):
            return lerp(760, 150, clamp((p - 50) / 34))
        melody_ribbon(c, t, head[0], -520, y_of, back=3.2, color="#7afcff" if power < 0.5 else "#ffd21f", width=7,
                      min_pitch=55)
        solo_bolts(c, x, head, BLOB_X, by, br, big=1.0 + 0.5 * power)
        if idx == 3:
            K.crowd(c, -40, W + 40, H + 10, t, x.b, n=40, size=70, seed=3)
        c.restore()
        if idx == 0:
            k = ease_back(clamp(first / 0.2), 3)
            fade = clamp((A.bar_time(33) - t) / 0.4)
            if fade > 0:
                c.save()
                c.translate(W / 2, 250)
                c.rotate(-5)
                c.scale(k, k)
                c.drawRect(skia.Rect.MakeLTRB(-900, -110, 900, 110), P("#ffd21f", 0.9 * fade))
                c.drawRect(skia.Rect.MakeLTRB(-900, -110, 900, 110), P("#120a18", fade, stroke=12))
                FX.outlined_text(c, "GUITAR SOLO", 0, 60, 170, "RubikMonoOne.ttf", fill="#120a18", stroke="#ffffff",
                                 sw=10, a=fade, shadow=False)
                c.restore()
        if idx == 2:
            age = t - bar_t(36)
            if age < 2.0:
                c.save()
                c.translate(W / 2, H - 170)
                k = ease_back(clamp(age / 0.2), 3)
                c.scale(k, k)
                FX.outlined_text(c, "MAXIMUM VOLTAGE", 0, 50, 150, "RubikMonoOne.ttf", fill="#ffd21f",
                                 stroke="#120a18", sw=18, extra_stroke="#7afcff", glow="#7afcff",
                                 a=clamp((2.0 - age) / 0.4))
                c.restore()
        if idx == 3 and x.bar > 39.3:
            FX.sfx(c, "GULP!?", W / 2, 300, 220, t - A.bar_time(39.4), color="#ff5fa8", angle=-8, life=2.0)
        return post(bloom=0.5 + 0.2 * power, threshold=0.45, chroma=3 + 5 * x.snare + 4 * power,
                    flash=max(0.8 * math.exp(-first * 6) if idx == 0 else 0.0, 0.25 * hits),
                    flash_col="#dff8ff", impact=1.0 if (idx == 2 and t - bar_t(36) < 0.07) else 0.0)
    # idx 1: close, low angle on Zip shredding; fret hand follows the pitch
    cut = t - bar_t(34)
    cam(c, punch(x, 0.03), rot=-6 + 2 * math.sin(t))
    solo_backdrop(c, x, horizon=1200)
    def y_of(p):
        return lerp(820, 120, clamp((p - 50) / 34))
    pose = solo_play_pose(x, yaw=0.3, power=0.2)
    zx, zy, zs = 760, 1500, 1050
    head = to_screen(pose, zx, zy, zs, 0.66, 0.0)
    melody_ribbon(c, t, head[0], -700, y_of, back=2.5, color="#7afcff", width=9, min_pitch=55)
    c.drawRect(skia.Rect.MakeLTRB(300, 1150, 1300, 1200), P("#16102c"))
    zdraw(c, pose, zx, zy, zs, t, rim="#7afcff", rim_a=0.7)
    for n in recent_notes(t, 0.5, min_pitch=60):
        age = t - n[0]
        seed = int(n[0] * 1000)
        if age < 0.15:
            FX.lightning(c, head, (W + 100, 150 + hash1(seed) * 500), seed=seed, width=6, a=1 - age / 0.15, branches=2)
        if n[2] >= 72:
            FX.sfx(c, SFX_WORDS[seed % len(SFX_WORDS)], 1500, 250 + (seed % 3) * 140, 120, age, color="#ffd21f",
                   life=0.55, seed=seed)
    c.restore()
    FX.radial_lines(c, 400, 700, t, n=60, inner=0.5, color="#ffffff", a=0.3)
    return post(bloom=0.55, threshold=0.45, chroma=4 + 4 * x.snare, flash=0.6 * math.exp(-cut * 7), letterbox=0.35)


# ----------------------------------------------------------------------------
# INSIDE THE BLOB (bars 40-44)
# ----------------------------------------------------------------------------
def float_pose(x, expr="shock"):
    t = x.t
    p = Z.base_pose(0.3)
    p["guitar"] = "back"
    p["root"] = (0.0, -Z.PELVIS_H - 0.02)
    p["feet"] = {1: (0.12, -0.06 + 0.03 * math.sin(t * 1.3)), -1: (-0.06, -0.12 + 0.03 * math.sin(t * 1.1 + 1))}
    p["hands"] = {1: (0.27, -0.92 + 0.04 * math.sin(t * 1.2)), -1: (-0.3, -0.88 + 0.04 * math.sin(t * 1.5))}
    p["hand_shape"] = {1: "open", -1: "open"}
    p["elbow_hint"] = {1: (0.5, 1.0), -1: (-0.5, 1.0)}
    p["hair_v"] = (0.02 * math.sin(t), -0.22)
    p["scarf_wind"] = (0.3 * math.sin(t * 0.7), -1.0)
    p["expr"] = expr
    return p


def s_inside(c, x):
    t = x.t
    first = t - bar_t(40)
    bar = x.bar
    Wd.blob_dimension(c, t, x.b)
    vhs = clamp((bar - 42) / 0.4) * (1 - clamp((bar - 43.2) / 0.6))
    Wd.floaty_items(c, t, vhs_focus=vhs)
    expr = "shock"
    if bar >= 42:
        expr = "shock" if bar < 42.5 else "soft"
    if bar >= 43:
        expr = "soft"
    if bar >= 44:
        expr = "grin"
    if bar < 44.3:
        pose = float_pose(x, expr)
        c.save()
        c.translate(W / 2, H / 2 + 120)
        c.rotate(12 * math.sin(t * 0.6))
        c.translate(-W / 2, -(H / 2 + 120))
        zdraw(c, pose, W / 2, H / 2 + 330 + 20 * math.sin(t * 0.9), 560, t, rim="#ffffff", rim_a=0.5)
        c.restore()
    else:
        pose = Z.play_pose(x.b, t, yaw=0.3, fret=0.5, note_age=note_age(t, 50))
        pose["expr"] = "grin"
        pose["eye_glow"] = clamp((bar - 44.3) / 0.3)
        pose["aura"] = 1.0
        pose["hair_v"] = (0.0, -0.15)
        zdraw(c, pose, W / 2, H / 2 + 360, 580, t, rim="#fff39a", rim_a=0.8)
    # cracks of light spreading from the edges on bar 44
    crack = clamp((bar - 44.2) / 0.6)
    if crack > 0:
        rng = np.random.default_rng(12)
        for k in range(16):
            ang = k / 16 * TAU
            x0, y0 = W / 2 + math.cos(ang) * 1300, H / 2 + math.sin(ang) * 800
            pts = [(x0, y0)]
            px, py = x0, y0
            for j in range(int(10 * crack)):
                a2 = math.atan2(H / 2 - py, W / 2 - px) + rng.uniform(-0.7, 0.7)
                px, py = px + math.cos(a2) * 70, py + math.sin(a2) * 70
                pts.append((px, py))
            path = path_from(pts, close=False)
            c.drawPath(path, P("#fffbe0", 0.6, stroke=18, blur=8))
            c.drawPath(path, P("#ffffff", 1.0, stroke=5))
    for li in (13, 14, 15, 16):
        float_text(c, L.lines()[li], t, W / 2, 170, size=92)
    if first < 0.8:
        FX.sfx(c, "GULP!", W / 2, H / 2, 260, first, color="#ff5fa8", life=0.8)
    stop = A.beat_time(44 * 4 + 3) + 0.1
    age = t - stop
    iris = 1.0 - ease_in(clamp(1 - first / 0.35), 2) if first < 0.35 else 1.0
    return post(bloom=0.35, threshold=0.6, sat=1.05, grain=0.07, vignette=0.55,
                flash=max(0.7 * math.exp(-first * 5), 0.9 * clamp(age / 0.3) if age > 0 else 0.0), flash_col="#fff6ff",
                impact=1.0 if 0 <= age < 0.08 else 0.0, fade=0.0, glitch=0.0)


# ----------------------------------------------------------------------------
# FINAL CHORUS AT DAWN (bars 45-58)
# ----------------------------------------------------------------------------
def dawn_sky(c, k, top=-300, bottom=H):
    Wd.sky(c, [(0, mix("#10103a", "#3a5ab8", k)), (0.45, mix("#4a2a7a", "#f07aa0", k)), (0.75, mix("#a04a7a", "#ffb37a", k)),
               (1, mix("#ff8a6a", "#fff0c0", k))], top, bottom)


def fireworks(c, x, strength=1.0, every=2, y_range=(120, 420)):
    cols = ["#ffd35a", "#7afcff", "#ff7ac0", "#ffffff", "#b8ff7a"]
    for j in range(-5, 1):
        bb = (x.bi // every + j) * every
        tb = A.beat_time(bb)
        age = x.t - tb
        if age < 0:
            continue
        seed = bb * 7 + 3
        fx_ = 250 + hash1(seed) * (W - 500)
        fy_ = y_range[0] + hash1(seed + 1) * (y_range[1] - y_range[0])
        if age < 0.3:
            mvp.rocket(c, fx_ + 60, H, fx_, fy_, age / 0.3, a=strength)
        else:
            mvp.firework(c, fx_, fy_, age - 0.3, seed=seed, color=cols[(bb // every) % len(cols)],
                         shape=["burst", "burst", "eye"][(bb // every) % 3] if False else "burst", a=strength, size=1.1, n=60)


def s_final(c, x):
    t = x.t
    bar = x.bar
    k_dawn = clamp((bar - 45) / 12)
    first = t - bar_t(45)
    catch_t = A.bar_time(55)
    if bar < 49:
        # breakout + sky dive after the falling trophy
        cam(c, punch(x, 0.03), rot=-8 + 4 * math.sin(t * 0.7))
        dawn_sky(c, k_dawn * 0.6)
        Wd.stars(c, t, 1 - k_dawn, seed=45, n=300, h=700)
        # clouds rushing upward (we're diving)
        for k in range(10):
            yy = (k * 260 - t * 900) % (H + 600) - 300
            Wd.cloud(c, (k * 397) % W, yy, 420, 90, "#ffe0f0", 0.35, seed=k)
        # explosion at the start
        if first < 1.2:
            FX.shockwave(c, W / 2, H / 2, first, rmax=1500, color="#ffffff", width=80, life=0.9)
            rng = np.random.default_rng(45)
            for i in range(120):
                ang = rng.uniform(0, TAU)
                sp = rng.uniform(300, 1400)
                d = sp * first
                px, py = W / 2 + math.cos(ang) * d, H / 2 + math.sin(ang) * d + 300 * first * first
                c.save()
                c.translate(px, py)
                c.rotate(first * 400 + i * 30)
                c.drawRect(skia.Rect.MakeLTRB(-10, -6, 10, 6), P(["#ff5fa8", "#ffc2e0", "#ffd21f", "#7afcff"][i % 4], clamp(1.2 - first)))
                c.restore()
        # trophy tumbling below, blob dizzy drifting away
        tf = clamp(first / 1.0)
        tx = lerp(W / 2, 1250, ease_out(tf)) + 60 * math.sin(t * 1.3)
        ty = lerp(H / 2, 700, ease_out(tf)) + 30 * math.sin(t * 2.3)
        K.draw_trophy(c, tx, ty, 170, t, shine=1.0, spin=t * 6)
        K.draw_blob(c, lerp(W / 2, 250, ease_out(tf)), lerp(H / 2, 250, ease_out(tf)), 70, t, mood="dizzy", look=(0, 0))
        # Zip diving on his lightning
        zx, zy = lerp(W / 2, 780, ease_out(tf)), lerp(H / 2 + 100, 460, ease_out(tf))
        c.save()
        c.translate(zx, zy)
        c.rotate(-35)
        c.translate(-zx, -zy)
        FX.lightning(c, (zx - 40, zy), (zx - 900, zy - 500), seed=int(t * 20), width=9, a=0.9, branches=2)
        pose = surf_pose(x, yaw=0.9)
        pose["hands"] = {1: (0.4, -0.7), -1: (-0.35, -0.8)}
        pose["expr"] = "determined"
        zdraw(c, pose, zx, zy, 380, t, rim="#fff0c0", rim_a=0.6)
        c.restore()
        c.restore()
        FX.hlines(c, t, n=24, color="#ffffff", a=0.3, speed=3000, seed=8)
        if bar < 47:
            mega_text(c, L.lines()[17], t, W / 2, 330, size=230, pal="pink", kick=x.kick)
        else:
            slam_text(c, L.lines()[18], t, W / 2, 230, size=100, pal="hot", maxw=1700, kick=x.kick)
        return post(bloom=0.55, threshold=0.45, impact=1.0 if first < 0.07 else 0.0, flash=0.95 * math.exp(-first * 4),
                    chroma=3 + 4 * x.snare)
    if bar < 53:
        # rooftops at dawn, crowds cheering, fireworks on the downbeats
        scroll = 1500 * (t - bar_t(49))
        cam(c, punch(x, 0.03))
        dawn_sky(c, 0.4 + 0.4 * k_dawn, bottom=900)
        Wd.stars(c, t, 0.3, seed=49, n=200, h=400)
        fireworks(c, x)
        Wd.city(c, "far", scroll * 0.15, 800, t, scale=1.0, a=1.0)
        Wd.city(c, "mid", scroll * 0.4, 950, t, scale=1.0)
        for k in range(3):
            X = (k * 800 - scroll * 0.9) % (W + 800) - 400
            c.drawRect(skia.Rect.MakeLTRB(X, 880, X + 620, H + 50), P("#1a1030"))
            K.crowd(c, X + 20, X + 600, 890, t, x.b, n=10, size=64, seed=k + 5, cheer=1.2)
        tx = 1350 + 80 * math.sin(t * 1.1)
        ty = 520 + 120 * fract((t - bar_t(49)) * 0.23)
        K.draw_trophy(c, tx, ty, 150, t, shine=1.0, spin=t * 5)
        zx, zy = 640, 600 + 40 * math.sin(t * 1.4)
        FX.lightning(c, (zx - 60, zy + 5), (zx - 1100, zy + 80), seed=int(t * 20), width=10, a=0.95)
        pose = surf_pose(x)
        pose["hands"] = {1: (0.42, -0.8), -1: (-0.4, -0.72)}
        pose["hand_shape"] = {1: "open", -1: "horns"}
        zdraw(c, pose, zx, zy, 400, t, rim="#fff0c0", rim_a=0.6)
        c.restore()
        FX.hlines(c, t, n=20, color="#ffffff", a=0.3, speed=3200, seed=9)
        slam_text(c, L.lines()[19 if bar < 51 else 20], t, W / 2, 230, size=112, pal="hot" if bar < 51 else "pink",
                  maxw=1700, kick=x.kick)
        return post(bloom=0.5, threshold=0.5, chroma=2 + 3 * x.snare, flash=0.5 * math.exp(-(t - bar_t(49)) * 6))
    if t < catch_t:
        # slow-motion reach
        slow = t - bar_t(53)
        cam(c, 1.0 + 0.1 * slow / 4.3)
        dawn_sky(c, 0.85)
        rng = np.random.default_rng(53)
        for i in range(60):
            px = (rng.uniform(0, W) - slow * 60 * rng.uniform(0.5, 1.5)) % W
            py = (rng.uniform(0, H) + slow * 20) % H
            c.drawCircle(px, py, rng.uniform(2, 6), P("#fff6d0", 0.6))
        tx, ty = 1380 - slow * 40, 430 + slow * 25
        K.draw_trophy(c, tx, ty, 260, t, shine=1.0, spin=0.2 * math.sin(slow))
        zx, zy = 520 + slow * 70, 760
        c.save()
        c.translate(zx, zy)
        c.rotate(-62)
        c.translate(-zx, -zy)
        p = Z.base_pose(0.85)
        p["guitar"] = "back"
        p["hands"] = {1: (0.55, -0.62), -1: (-0.25, -0.5)}
        p["hand_shape"] = {1: "open", -1: "fist"}
        p["elbow_hint"] = {1: (0.0, -1.0), -1: (-1.0, 0.0)}
        p["feet"] = {1: (0.05, 0.0), -1: (-0.2, -0.05)}
        p["expr"] = "determined"
        p["hair_v"] = (-0.25, 0.0)
        p["speed"] = 0.4
        zdraw(c, p, zx, zy + 300, 720, t, rim="#fff0c0", rim_a=0.7)
        c.restore()
        c.restore()
        FX.radial_lines(c, tx, ty, t, n=90, inner=0.25, color="#ffffff", a=0.45, fps_hold=6)
        slam_text(c, L.lines()[21], t, W / 2, 170, size=104, pal="gold", maxw=1700, shake_amt=2)
        return post(bloom=0.5, threshold=0.5, sat=1.12, letterbox=0.5, flash=0.4 * math.exp(-slow * 6))
    if bar < 56:
        # CATCH
        age = t - catch_t
        sx_, sy_ = shake(x, 30 * math.exp(-age * 4))
        cam(c, 1.05 + 0.05 * math.exp(-age * 3), ox=sx_, oy=sy_)
        c.drawRect(skia.Rect.MakeLTRB(-200, -200, W + 200, H + 200), P(shader=rad_grad((W / 2, 400), W, [(0, "#fff6c0", 1), (0.5, "#ffb37a", 1), (1, "#c04a7a", 1)])))
        FX.burst(c, W / 2, 420, 1400, n=24, color="#fff0a0", a=0.5, inner=0.2, rot=t * 0.3)
        FX.shockwave(c, W / 2, 420, age, rmax=1500, color="#ffffff", width=70, life=0.8)
        p = Z.base_pose(0.3)
        p["guitar"] = "back"
        p["hands"] = {1: (0.1, -1.12), -1: (-0.2, -0.62)}
        p["hand_shape"] = {1: "open", -1: "fist"}
        p["elbow_hint"] = {1: (1.0, 0.0), -1: (-1.0, 0.3)}
        p["expr"] = "grin"
        p["hair_v"] = (0.0, -0.1)
        p["feet"] = {1: (0.12, -0.05), -1: (-0.1, -0.12)}
        zdraw(c, p, W / 2, 1180, 820, t, rim="#ffffff", rim_a=0.6)
        J = Z.joints(p)
        hx, hy = J[("wrist", 1)]
        K.draw_trophy(c, W / 2 + hx * 820, 1180 + hy * 820 - 20, 240, t, shine=1.0)
        c.restore()
        mega_text(c, L.lines()[22], t, W / 2, 900, size=210, pal="gold", kick=x.kick)
        return post(bloom=0.6, threshold=0.45, impact=1.0 if age < 0.07 else 0.0, flash=0.95 * math.exp(-age * 5),
                    chroma=5 * math.exp(-age * 3))
    # bars 56-58: hero landing at sunrise, friends arrive
    lt = t - bar_t(56)
    cam(c, punch(x, 0.025) * (1.0 + 0.04 * math.exp(-lt * 4)))
    dawn_sky(c, 1.0, bottom=820)
    Wd.sun_disc = None
    Wd.glow(c, 1450, 760, 700, "#ffd08a", 0.7)
    c.drawCircle(1450, 780, 200, P("#fff2c0"))
    Wd.city(c, "far", 2000, 830, t, scale=1.0, a=0.9, beacon=False)
    Wd.city(c, "mid", 2600, 920, t, scale=1.0, beacon=False)
    fireworks(c, x, strength=0.6, every=4, y_range=(100, 300))
    c.drawRect(skia.Rect.MakeLTRB(-100, 900, W + 100, H + 100), P("#241a3a"))
    c.drawRect(skia.Rect.MakeLTRB(-100, 890, W + 100, 912), P("#3a2a5a"))
    FX.shockwave(c, 760, 900, lt, rmax=900, color="#ffffff", width=40, life=0.6)
    land = ease_out(clamp(lt / 0.25))
    if lt < 1.0:
        p = Z.base_pose(0.7)
        p["guitar"] = "back"
        p["root"] = (0.0, -0.2)
        p["feet"] = {1: (0.22, 0.0), -1: (-0.3, 0.0)}
        p["hands"] = {1: (0.05, -0.03), -1: (-0.12, -0.95)}
        p["hand_shape"] = {1: "fist", -1: "open"}
        p["lean"] = 28
        p["expr"] = "determined"
        zy = lerp(-300, 900, land)
        zdraw(c, p, 760, zy, 560, t, rim="#fff0c0", rim_a=0.6)
        J = Z.joints(p)
        hx, hy = J[("wrist", -1)]
        K.draw_trophy(c, 760 + hx * 560, zy + hy * 560 - 10, 160, t, shine=1.0)
        if lt < 0.6:
            FX.sfx(c, "KA-THOOM!", 900, 700, 160, lt - 0.2, color="#ffd21f", life=0.8)
    else:
        p = Z.base_pose(0.35)
        p["guitar"] = "back"
        p["hands"] = {1: (0.12, -1.1), -1: (-0.28, -0.52)}
        p["hand_shape"] = {1: "open", -1: "horns"}
        p["elbow_hint"] = {1: (1.0, 0.0), -1: (-1.0, 0.3)}
        p["expr"] = "grin"
        p["root"] = (0.0, -Z.PELVIS_H + 0.015 * math.exp(-x.u * 5))
        zdraw(c, p, 760, 900, 560, t, rim="#fff0c0", rim_a=0.6)
        J = Z.joints(p)
        hx, hy = J[("wrist", 1)]
        K.draw_trophy(c, 760 + hx * 560, 900 + hy * 560 - 10, 170, t, shine=1.0)
    # AMPY rockets in, the (reformed) blob hops over, both cheering
    ampy_in = clamp((t - A.bar_time(57)) / 0.6)
    if ampy_in > 0:
        ax = lerp(W + 200, 1180, ease_out(ampy_in))
        ay = lerp(300, 900, ease_out(ampy_in))
        hop = abs(math.sin(math.pi * x.u)) if ampy_in >= 1 else 0.0
        if ampy_in < 1:
            for q in range(5):
                c.drawCircle(ax + 20 + q * 20, ay + 10 + q * 12, 30 - q * 5, P(["#fff39a", "#ffb31a", "#ff5a1f"][q % 3], 0.8))
        K.draw_ampy(c, ax, ay, 190, t, bass=x.bass, mood="happy", hop=hop)
    blob_in = clamp((t - A.bar_time(57.5)) / 0.6)
    if blob_in > 0:
        bx = lerp(-150, 420, ease_out(blob_in))
        hop = abs(math.sin(math.pi * x.u))
        K.draw_blob(c, bx, 900 - 60 - 100 * hop, 64, t, mood="happy", look=(1, 0), sy=1 + 0.1 * hop)
    K.crowd(c, -40, W + 40, H + 30, t, x.b, n=36, size=80, seed=11, cheer=1.3)
    mvp.confetti(c, t, seed=5, n=140, t0=bar_t(56))
    c.restore()
    slam_text(c, L.lines()[23], t, W / 2, 220, size=104, pal="gold", maxw=1750, kick=x.kick)
    return post(bloom=0.5, threshold=0.5, sat=1.15, flash=0.7 * math.exp(-lt * 5), chroma=2 * x.snare)


# ----------------------------------------------------------------------------
# VICTORY JAM (bars 59-62)
# ----------------------------------------------------------------------------
def s_victory(c, x):
    t = x.t
    idx, sb = x.shot([59, 60, 61, 62])
    first = t - A.bar_time([59, 60, 61, 62][idx])
    zoom = [1.0, 1.25, 1.55, 1.05][idx]
    cx_, cy_ = [(W / 2, H / 2), (W / 2 + 60, 640), (720, 520), (W / 2, H / 2)][idx]
    cam(c, zoom * punch(x, 0.03), cx=cx_, cy=cy_, rot=[0, -3, 4, 0][idx])
    dawn_sky(c, 1.0, bottom=820)
    Wd.glow(c, 1500, 700, 800, "#ffd08a", 0.75)
    c.drawCircle(1500, 700, 190, P("#fff4d0"))
    for k in range(6):
        bx_ = (k * 260 + t * 70) % (W + 400) - 200
        by_ = 200 + (k % 3) * 60 + 20 * math.sin(t * 2 + k)
        f = math.sin(t * 9 + k) * 10
        pth = skia.Path()
        pth.moveTo(bx_ - 16, by_ - f)
        pth.quadTo(bx_ - 6, by_ - 4, bx_, by_)
        pth.quadTo(bx_ + 6, by_ - 4, bx_ + 16, by_ - f)
        c.drawPath(pth, P("#3a2040", 0.8, stroke=3))
    fireworks(c, x, strength=0.55, every=4, y_range=(90, 280))
    Wd.city(c, "far", 2100, 830, t, scale=1.0, beacon=False, a=0.9)
    Wd.city(c, "mid", 2700, 930, t, scale=1.0, beacon=False)
    c.drawRect(skia.Rect.MakeLTRB(-200, 900, W + 200, H + 200), P("#241a3a"))
    c.drawRect(skia.Rect.MakeLTRB(-200, 890, W + 200, 912), P("#3a2a5a"))
    K.draw_trophy(c, 1650, 890, 150, t, shine=1.0)
    # AMPY + blob dance on the beat
    hop = abs(math.sin(math.pi * x.u))
    K.draw_ampy(c, 1180, 900, 200, t, bass=x.bass, mood="happy", hop=hop, lean=8 * math.sin(math.pi * x.b / 2))
    K.draw_blob(c, 380, 900 - 64 * 0.9 - 110 * hop, 64, t, mood="happy", look=(1, 0), sy=1 + 0.12 * hop,
                sx=1 - 0.08 * hop)
    jump = 0.0
    if idx == 3:
        jump = ease_out(clamp((x.b - (62 * 4 + 2.5)) / 1.5))
    pose = Z.play_pose(x.b, t, yaw=0.3, fret=Z.pitch_to_fret(A.pitch_at(t)), note_age=note_age(t, 50))
    pose["expr"] = "grin" if int(x.b) % 4 == 3 else "rock"
    if jump > 0:
        pose["feet"] = {1: (0.16, -0.2 * jump), -1: (-0.14, -0.1 * jump)}
    zdraw(c, pose, 760, 900 - 260 * jump, 560, t, rim="#fff0c0", rim_a=0.6)
    K.crowd(c, -40, W + 40, H + 40, t, x.b, n=36, size=80, seed=21, cheer=1.3)
    mvp.confetti(c, t, seed=9, n=110, t0=bar_t(59) - 3)
    c.restore()
    for li in (24, 25):
        sky_text(c, L.lines()[li], t, W / 2, 190, size=150)
    return post(bloom=0.45, threshold=0.52, sat=1.15, flash=0.45 * math.exp(-first * 7), chroma=1.5 * x.snare)


# ----------------------------------------------------------------------------
# ENDING: final chord freeze, back to the webcam
# ----------------------------------------------------------------------------
def s_end(c, x):
    t = x.t
    t0 = bar_t(63)
    lt = t - t0
    if t < 161.9:
        z = 1.0 + 0.08 * lt
        cam(c, z, cy=H / 2 - 40)
        dawn_sky(c, 1.0, bottom=900)
        Wd.glow(c, W / 2, 560, 900, "#fff0b0", 0.9)
        c.drawCircle(W / 2, 560, 300, P("#fffbe8"))
        FX.burst(c, W / 2, 560, 1500, n=28, color="#ffe8a0", a=0.35, inner=0.25, rot=lt * 0.1)
        p = Z.base_pose(0.3)
        p["guitar"] = "raise"
        p["gtr_ang"] = -75
        p["feet"] = {1: (0.2, -0.24), -1: (-0.2, -0.06)}
        p["foot_ang"] = {1: -25, -1: 30}
        p["hands"] = {1: (0.12, -1.05), -1: (0.04, -1.0)}
        p["hand_shape"] = {1: "fist", -1: "fist"}
        p["expr"] = "rock"
        p["hair_v"] = (0.0, -0.2)
        ring = math.exp(-lt * 1.5)
        sx_, sy_ = shake(x, 12 * ring)
        c.translate(sx_, sy_)
        zdraw(c, p, W / 2 + 40, 1020, 760, t, sil="#1a0e2a" if lt < 0.05 else None, rim="#ffffff", rim_a=0.8)
        c.restore()
        FX.radial_lines(c, W / 2, 560, t, n=110, inner=0.3, color="#ffffff", a=0.45 * ring + 0.1, fps_hold=8)
        if lt > 0.3:
            k = ease_back(clamp((lt - 0.3) / 0.25), 2.5)
            c.save()
            c.translate(W / 2, 150)
            c.scale(k, k)
            FX.outlined_text(c, "CHASE", 0, 70, 190, "RubikMonoOne.ttf", fill="#ffd21f", stroke="#120a18", sw=24,
                             extra_stroke="#2fe3d7", glow="#2fe3d7")
            c.restore()
        return post(bloom=0.55, threshold=0.45, impact=1.0 if lt < 0.08 else 0.0, flash=0.9 * math.exp(-lt * 4),
                    sat=1.15, glitch=0.9 * clamp((t - 161.6) / 0.3))
    # back in the room: the trophy home, the blob now a friend on the shelf
    rt = t - 161.9
    cam(c, 1.0, ox=vnoise(t * 0.7, 3) * 6, oy=vnoise(t * 0.6, 4) * 5)
    Wd.room(c, t, light=1.0, amp_glow=0.4, bass=0.0, trophy=True, guitar_on_stand=True)
    K.draw_blob(c, 300, 560 - 48 * 0.9, 48, t, mood="happy", look=(1, 0.3), arms_up=0.4 * abs(math.sin(t * 4)))
    K.draw_ampy(c, 1650, 560, 120, t, mood="happy", hop=abs(math.sin(t * 3)) * 0.3)
    p = Z.base_pose(0.25)
    p["guitar"] = "none"
    p["talk"] = 0.0
    bub = None
    for (a0, a1, text) in L.OUTRO_BUBBLES:
        if a0 <= t < a1 + 0.4:
            bub = (a0, a1, text)
    if bub:
        a0, a1, text = bub
        # mouth flaps while the (silent) line types on
        prog = clamp((t - a0) / ((a1 - a0) * 0.6))
        p["talk"] = (0.3 + 0.5 * abs(math.sin(t * 18))) if prog < 1 else 0.0
    p["expr"] = "smug"
    if t > 165.4:
        p["expr"] = "wink"
        p["hands"][-1] = (0.24, -0.72)
        p["hand_shape"][-1] = "point"
        p["elbow_hint"][-1] = (0.2, 1.0)
        p["arm_front"][-1] = True
    zdraw(c, p, 760, 1530, 1180, t, rim="#b86bff", rim_a=0.35)
    c.restore()
    if bub:
        a0, a1, text = bub
        bubble(c, text, 1030, 250, (850, 420), t - a0, clamp((t - a0) / ((a1 - a0) * 0.6)), size=66, maxw=560)
    chat = [(162.4, "voltkid", "GG ZIP", "#2fe3d7"), (163.3, "moth_lamp", "the blob is so cute", "#ff9ad0"),
            (164.2, "bytebass", "that solo tho", "#9aff7a"), (165.9, "zip_fan99", "rude (affectionate)", "#ffd21f"),
            (166.8, "cheesecat", "2000 NEXT", "#ffb35a")]
    Wd.webcam_ui(c, t, L.CHAT[-4:] + chat, rec_t=t - 160)
    card = clamp((t - 167.4) / 0.6)
    if card > 0:
        c.drawRect(skia.Rect.MakeLTRB(0, 0, W, H), P("#0a0614", 0.55 * card))
        k = ease_back(clamp((t - 167.4) / 0.35), 2)
        c.save()
        c.translate(W / 2, H / 2 - 40)
        c.scale(k, k)
        FX.outlined_text(c, "THANKS FOR 200", 0, 40, 122, "RubikMonoOne.ttf", fill="#ffd21f", stroke="#120a18", sw=18,
                         extra_stroke="#ff3d8a", glow="#ff3d8a", a=card)
        FX.outlined_text(c, "see you at 2,000", 0, 170, 80, "PermanentMarker.ttf", fill="#ffffff", stroke="#120a18",
                         sw=10, a=card)
        c.restore()
    fade = clamp((t - 169.2) / 0.9)
    return post(bloom=0.35, threshold=0.55, glitch=0.8 * clamp(1 - rt / 0.3), fade=fade, scan=0.05)
