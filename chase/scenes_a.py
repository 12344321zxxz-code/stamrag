"""Cold open (webcam roast), title drop, rooftop chase."""
import math

import numpy as np
import skia

from . import cast as K
from . import fxk as FX
from . import hero as Z
from . import lyrics as L
from . import world as Wd
from .core import (BM, H, TAU, W, C, P, clamp, ease_back, ease_in, ease_io, ease_out, font, fract, hash1, inv_lerp,
                   lerp, lin_grad, mix, path_from, poly, pulse, rad_grad, scale_c, smooth, vnoise)
from .timeline import (A, bar_t, bubble, cam, float_text, fly_text, mega_text, post, punch, shake, slam_text, zdraw)

ZX, ZY, ZS = 760, 1530, 1180          # Zip's placement in the webcam shots
CRINGE_T = 12.55                      # the moment the word lands in the bubble
DROP_T = 13.15
MERGE_T = 13.85
FLOOR = 905


def _phrase_idx(t):
    for i, (a, b) in enumerate(A.phrases):
        if a <= t < b + 0.35:
            return i
    return -1


def _talk(x):
    s = x.speech
    return 0.0 if s < 0.22 else clamp(0.25 + 0.9 * s) * (0.75 + 0.25 * math.sin(x.t * 40))


def cold_zip_pose(x):
    t = x.t
    p = Z.base_pose(0.3)
    p["guitar"] = "none"
    p["root"] = (0.0, -Z.PELVIS_H + 0.006 * math.sin(t * 2.2))
    p["feet"] = {1: (0.08, 0.0), -1: (-0.08, 0.0)}
    ph = _phrase_idx(t)
    p["talk"] = _talk(x) if ph >= 0 else 0.0
    p["blink"] = 1.0 if fract(t / 3.1) > 0.965 else 0.0
    look_away = t < 2.45
    p["head_yaw"] = lerp(0.85, 0.12, ease_io(clamp((t - 2.2) / 0.35))) if t < 2.6 else 0.12
    p["expr"] = "neutral"
    if look_away:
        p["expr"] = "deadpan"
    if ph == 0:
        p["expr"] = "neutral"
        p["head_tilt"] = -6
    elif ph == 1:
        p["expr"] = "deadpan"
        p["hands"][-1] = (0.24, -0.66)
        p["hand_shape"][-1] = "point"
        p["elbow_hint"][-1] = (0.2, 1.0)
        p["arm_front"][-1] = True
    elif ph == 2:
        p["expr"] = "disgust"
        p["hands"][1] = (-0.3, -0.8)
        p["hand_shape"][1] = "point"
        p["elbow_hint"][1] = (0.2, 1.0)
        p["head_yaw"] = 0.05
        p["head_tilt"] = 5
    elif ph == 3:
        p["expr"] = "smug"
        p["lean"] = 14
        p["head_tilt"] = -4
        p["hands"][-1] = (0.14, -0.55)
        p["hand_shape"][-1] = "open"
        p["elbow_hint"][-1] = (0.0, 1.0)
    elif ph == 4:
        p["expr"] = "smug"
        p["blink"] = 1.0
        p["lean"] = -6
        for sd in (-1, 1):
            p["hands"][sd] = (0.07 * -sd, -0.63)
            p["hand_shape"][sd] = "fist"
            p["elbow_hint"][sd] = (sd * 1.0, 0.4)
            p["arm_front"][sd] = True
    elif ph == 5:
        p["expr"] = "disgust" if t < CRINGE_T + 0.2 else "smug"
        p["head_tilt"] = 8 * math.sin(t * 30) * clamp(1 - abs(t - CRINGE_T - 0.2) * 3)  # shiver on the word
        p["hands"][-1] = (0.2, -0.95)
        p["hand_shape"][-1] = "open"
        p["elbow_hint"][-1] = (0.5, 1.0)
        p["arm_front"][-1] = True
    elif ph == 6:
        p["expr"] = "smug"
        p["head_yaw"] = 0.3
    # guitar grabbed at ~14.75, then plug in, chord, dive, hold
    if t >= 14.75:
        b = x.b
        age = x.t - 17.0
        p2 = Z.play_pose(b, t, yaw=0.32, fret=0.55, strum_rate=0 if t < 17.0 else 1, intensity=0.0 if t < 17 else 1.0,
                         note_age=age if age > 0 else 9.0)
        for k in ("talk", "blink", "head_yaw"):
            p2[k] = p[k]
        p2["expr"] = p["expr"] if t < 16.3 else "rock"
        p2["root"] = p["root"]
        p2["feet"] = p["feet"]
        p2["strum"] = 0.8 if t < 17.0 else math.cos(clamp(age / 0.12) * math.pi)
        if 17.0 <= t < 19.47:
            p2["expr"] = "rock"
            p2["gtr_glow"] = clamp(1 - (t - 17.0) / 2.5)
            p2["string_vib"] = 1.0
            p2["hair_bounce"] = 0.5 * math.exp(-(t - 17.0) * 4)
        if 19.47 <= t < 22.0:
            dm = A.dive(t)
            dip = clamp((44.8 - dm) / 10.0)
            p2["lean"] = -4 - 16 * dip
            p2["expr"] = "rock"
            p2["blink"] = 1.0 if dip > 0.3 else 0.0
            p2["head_tilt"] = -12 * dip
            p2["gtr_glow"] = 0.5 + 0.5 * dip
            p2["string_vib"] = 1.0
        if t >= 22.0:
            k = clamp((t - 22.0) / 1.2)
            p2["shade_eyes"] = k
            p2["eye_glow"] = clamp((t - 22.6) / 0.6)
            p2["expr"] = "focus"
            p2["gtr_glow"] = 0.3 + 0.7 * k
            p2["aura"] = 0.6 * k
        p = p2
    return p


def blob_state(t):
    """Position/size/mood of the newborn blob during the cold open (room coords)."""
    if t < MERGE_T:
        return None
    r = 70 * ease_back(clamp((t - MERGE_T) / 0.35), 2.2)
    x, y = 560.0, FLOOR - r * 0.9
    mood = "shock" if t < 14.4 else "panic"
    look = (math.sin(t * 5), 0.0) if t < 15.5 else (1.0, 0.0)
    arms = 0.0
    trophy = False
    if 15.5 <= t < 16.4:
        f = ease_io((t - 15.5) / 0.9)
        x = lerp(560, 470, f)
        y = lerp(FLOOR - r * 0.9, 560 - r * 0.9, f) - math.sin(f * math.pi) * 260
        mood = "smug"
    elif t >= 16.4:
        x, y = 470.0, 560 - r * 0.9
        mood = "smug"
        look = (-1.0, 0.0) if t < 22.0 else (0.6, -0.3)
    if t >= 22.2:
        arms = ease_out(clamp((t - 22.2) / 0.4))
        trophy = True
        mood = "smug"
    return dict(x=x, y=y, r=r, mood=mood, look=look, arms=arms, trophy=trophy)


def _cringe_word(c, x, t):
    """The word CRINGE: pops out of the bubble, drops, bounces, then melts into the blob."""
    if t < CRINGE_T:
        return
    word = "CRINGE"
    size = 150
    f = font("Bangers.ttf", size)
    x0 = 1010 - f.measureText(word) / 2
    y0 = 520
    xs = []
    xx = x0
    for ch in word:
        w = f.measureText(ch)
        xs.append((xx + w / 2, w))
        xx += w * 1.02
    for i, ch in enumerate(word):
        cx, w = xs[i]
        cy = y0
        rot = 0.0
        sc = ease_back(clamp((t - CRINGE_T) / 0.18), 3.0)
        jig = math.sin(t * 40 + i * 1.3) * 5 * clamp(1 - (t - CRINGE_T) * 0.8)
        a = 1.0
        ts = DROP_T + i * 0.045
        if t > ts:
            dt = t - ts
            g = 2600.0
            fall_h = FLOOR - 40 - cy
            tf = math.sqrt(2 * fall_h / g)
            if dt < tf:
                cy = cy + 0.5 * g * dt * dt
            else:
                bt = dt - tf
                amp = 90 * math.exp(-bt * 5)
                cy = FLOOR - 40 - abs(math.sin(bt * 12)) * amp
            cx = lerp(cx, 560 + (i - 2.5) * 30, ease_io(clamp(dt / 0.7)))
            rot = (i - 2.5) * 40 * clamp(dt / 0.6)
            if t > MERGE_T - 0.15:
                m = ease_in(clamp((t - (MERGE_T - 0.15)) / 0.4))
                cx = lerp(cx, 560, m)
                cy = lerp(cy, FLOOR - 60, m)
                sc *= (1 - m)
                a = 1 - m
        if a <= 0.01 or sc <= 0.01:
            continue
        c.save()
        c.translate(cx, cy + jig)
        c.rotate(rot)
        c.scale(sc, sc)
        c.drawString(ch, -w / 2 + 5, size * 0.35 + 8, f, P("#000000", 0.4 * a))
        c.drawString(ch, -w / 2, size * 0.35, f, P("#3a0a24", a, stroke=16))
        c.drawString(ch, -w / 2, size * 0.35, f, P("#ff5fa8", a))
        c.drawString(ch, -w / 2, size * 0.35, f, P("#ffffff", 0.6 * a, stroke=3))
        c.restore()
    # shiver lines around the word right after it lands
    if CRINGE_T < t < DROP_T:
        for k in range(6):
            ang = k / 6 * TAU + t * 3
            r0 = 250
            cxw = 1010
            c.drawLine(cxw + math.cos(ang) * r0, y0 + math.sin(ang) * r0 * 0.5, cxw + math.cos(ang) * (r0 + 40),
                       y0 + math.sin(ang) * (r0 + 40) * 0.5, P("#ff5fa8", 0.8, stroke=6))


def s_cold(c, x):
    t = x.t
    # camera: gentle handheld; push-ins on key lines; tilt with the whammy dive
    zoom = 1.0
    cx, cy = W / 2, H / 2
    rot = 0.0
    ph = _phrase_idx(t)
    if ph == 3:
        zoom = 1.0 + 0.12 * ease_io(clamp((t - A.phrases[3][0]) / 0.5))
        cx, cy = 900, 480
    if ph == 5 and t > CRINGE_T:
        zoom = 1.0 + 0.06 * ease_out(clamp((t - CRINGE_T) / 0.2))
    dip = 0.0
    if 19.47 <= t < 22.0:
        dip = clamp((44.8 - A.dive(t)) / 10.0)
        rot = -6 * dip
        cy = H / 2 + 60 * dip
    if t >= 22.0:
        zoom = 1.0 + 0.25 * ease_in(clamp((t - 22.0) / 1.4))
        cx, cy = ZX + 40, 470
    hx, hy = vnoise(t * 0.7, 3) * 8, vnoise(t * 0.6, 4) * 6
    chord = math.exp(-max(0.0, t - 17.0) * 3.0) if t >= 17.0 else 0.0
    sx_, sy_ = shake(x, 22 * chord)
    cam(c, zoom, cx, cy, rot, hx + sx_, hy + sy_)
    light = 1.0 if t < 22.0 else lerp(1.0, 0.35, clamp((t - 22.0) / 0.8))
    amp_glow = clamp((t - 16.2) / 0.8) * (0.6 + 0.4 * chord) if t > 16.2 else 0.0
    bs = blob_state(t)
    Wd.room(c, t, light=light, sag=dip, amp_glow=amp_glow, bass=chord, fairy=light,
            trophy=not (bs and bs["trophy"]), guitar_on_stand=t < 14.75, poster_flap=chord + dip)
    # blob (behind Zip when on the floor / shelf)
    if bs:
        K.draw_blob(c, bs["x"], bs["y"], bs["r"], t, look=bs["look"], mood=bs["mood"], arms_up=bs["arms"],
                    ground=bs["y"] + bs["r"] * 0.9, sy=1.0 - 0.25 * dip, sx=1.0 + 0.2 * dip)
        if bs["trophy"]:
            K.draw_trophy(c, bs["x"], bs["y"] - bs["r"] * (1.05 + 0.5 * bs["arms"]), 120, t, shine=1.0)
    pose = cold_zip_pose(x)
    zdraw(c, pose, ZX, ZY, ZS, t, rim="#b86bff", rim_a=0.35)
    # grab smear
    if 14.6 < t < 14.95:
        k = 1 - abs(t - 14.75) / 0.2
        for i in range(8):
            c.drawLine(900 + i * 40, 520 + i * 30, 1330, 700 + i * 20, P("#ffffff", 0.5 * k, stroke=10))
    # chord shockwave + SFX
    if t >= 17.0:
        FX.shockwave(c, ZX + 60, 900, t - 17.0, rmax=1400, color="#ffe06a", width=60, life=0.7)
        FX.sfx(c, "KRANG!!", 1200, 380, 210, t - 17.0, color="#ffd21f", angle=-10, life=1.8, seed=3)
    if t >= 16.2:
        FX.sfx(c, "click", ZX - 180, 1010, 70, t - 16.25, color="#ffffff", angle=8, life=0.7)
    # whammy dive: the glowing 'power line' follows the pitch
    if 19.3 <= t < 23.3:
        a_ = clamp((t - 19.3) / 0.2) * clamp((23.3 - t) / 0.3)
        pts = []
        for i in range(0, 41):
            xx = -100 + i * (W + 200) / 40
            # the line lags behind the live pitch from left to right
            tt = t - (1 - i / 40) * 0.5
            m = A.dive(tt) if 19.47 <= tt < 22.0 else 44.8
            yy = lerp(760, 540, clamp((m - 34.5) / 10.5)) + math.sin(i * 1.3 + t * 30) * 6
            pts.append((xx, yy))
        pth = path_from(pts, close=False, smooth_k=0.6)
        c.drawPath(pth, P("#7afcff", 0.45 * a_, stroke=40, blur=16))
        c.drawPath(pth, P("#bffcff", a_, stroke=8))
        if t < 22.2:
            word = "BWOOOOOWWW"
            fnt = font("Bangers.ttf", 110)
            for i, ch in enumerate(word):
                k = (i + 0.5) / len(word)
                px, py = pts[int(4 + k * 32)]
                app = clamp((t - 19.5 - i * 0.12) / 0.15)
                if app > 0:
                    c.save()
                    c.translate(px, py - 30)
                    c.rotate(math.sin(t * 8 + i) * 10)
                    c.scale(app, app * (1 + 0.6 * clamp((44.8 - A.dive(t)) / 10)))
                    c.drawString(ch, -30, 0, fnt, P("#0a1a2a", 0.9, stroke=14))
                    c.drawString(ch, -30, 0, fnt, P("#7afcff"))
                    c.restore()
    c.restore()
    # --- screen-space overlays -------------------------------------------------
    # speech bubbles (original lines)
    ph_show = _phrase_idx(t)
    if 0 <= ph_show < len(L.INTRO_BUBBLES) and t < 16.6:
        a0, a1 = A.phrases[ph_show]
        text = L.INTRO_BUBBLES[ph_show]
        prog = clamp((t - a0) / max(0.3, (a1 - a0) * 0.75))
        if ph_show == 5:
            text = "...that's kinda"
            prog = clamp((t - a0) / 0.8)
        bubble(c, text, 1030, 250, (ZX + 90, 420), t - a0, prog, size=66, maxw=560)
    _cringe_word(c, x, t)
    if bs and MERGE_T < t < 15.2:
        FX.sfx(c, "blorp!", 560, 760, 80, t - (MERGE_T + 0.1), color="#ff9ad0", angle=-12, life=1.0)
    # chat + webcam UI
    ui_a = 1.0 if t < 22.0 else lerp(1.0, 0.0, clamp((t - 22.0) / 0.5))
    Wd.webcam_ui(c, t, L.CHAT, alpha=ui_a)
    # the pick slide: a white slash across the screen
    pf = 0.0
    if 23.47 <= t < 24.2:
        f = clamp((t - 23.47) / 0.5)
        x0, y0 = lerp(W + 200, -200, ease_in(f)), lerp(-150, H + 150, ease_in(f))
        for k in range(3):
            w = [70, 30, 8][k]
            c.drawLine(W + 300, -250, x0, y0, P(["#7afcff", "#ffffff", "#ffffff"][k], [0.4, 0.9, 1][k], stroke=w, cap="round"))
        FX.sfx(c, "SKRRRT!", W / 2, H / 2, 200, t - 23.5, color="#ffffff", stroke="#101020", angle=-28, life=0.7)
    fade_in = 1 - clamp(t / 0.8)
    black = clamp((t - 24.05) / 0.12) if t < bar_t(0) else 0.0
    noise = clamp(1 - t / 1.2)
    return post(bloom=0.35 + 0.3 * chord, threshold=0.55, fade=max(fade_in, black), glitch=noise * 0.8,
                chroma=3 * chord + 4 * (t > 23.47 and t < 24.2), vignette=0.5 + 0.3 * (t > 22),
                flash=0.6 * chord if t < 17.3 else 0.0, flash_col="#fff6c8", scan=0.05)


# ----------------------------------------------------------------------------
# TITLE (bar 0)
# ----------------------------------------------------------------------------
def s_title(c, x):
    t = x.t
    t0 = bar_t(0)
    lt = t - t0
    b = x.b
    sx_, sy_ = shake(x, 30 * math.exp(-lt * 3) + 8 * x.kick)
    cam(c, 1.0 + 0.04 * math.exp(-lt * 4), ox=sx_, oy=sy_)
    Wd.room(c, t, light=0.9, amp_glow=1.0, bass=x.kick, trophy=False, guitar_on_stand=False, poster_flap=1.0)
    # the blob leaps out of the window with the trophy on beat 2
    f = clamp((b - 1.0) / 1.2)
    bx = lerp(470, 1480, ease_io(f))
    by = lerp(510, 330, f) - math.sin(f * math.pi) * 250
    br = 70 * (1 - 0.5 * f)
    if f < 1:
        K.draw_blob(c, bx, by, br, t, look=(1, -0.5), mood="panic", arms_up=1.0, sx=1 + 0.3 * f, sy=1 - 0.2 * f)
        K.draw_trophy(c, bx, by - br * 1.5, 120 * (1 - 0.4 * f), t, shine=1.0)
    # Zip: shocked, then launches toward the window
    p = Z.play_pose(b, t, yaw=0.35, fret=0.5, note_age=0.3)
    p["expr"] = "shock" if b < 2.4 else "angry"
    p["head_yaw"] = 0.55 if b > 1.2 else 0.3
    zx, zy, zs = ZX, ZY, ZS
    if b > 3.0:
        g = ease_in(clamp((b - 3.0) / 1.0), 2)
        zx = lerp(ZX, 1500, g)
        zy = lerp(ZY, 900, g)
        zs = lerp(ZS, 400, g)
        p = Z.run_pose(b, yaw=0.9, t=t)
        p["expr"] = "angry"
    zdraw(c, p, zx, zy, zs, t, rim="#b86bff", rim_a=0.3)
    c.restore()
    # CHASE: letters slam on the first five eighth notes
    word = "CHASE"
    size = 330
    f_ = font("RubikMonoOne.ttf", size)
    widths = [f_.measureText(ch) for ch in word]
    total = sum(widths) + 20 * 4
    xx = W / 2 - total / 2
    ty = H / 2 + 60
    for i, ch in enumerate(word):
        age = t - A.beat_time(i * 0.5)
        if age < 0:
            xx += widths[i] + 20
            continue
        k = ease_back(clamp(age / 0.12), 4.0)
        rot = (-8 + i * 4) * (1 - clamp(age / 0.2))
        c.save()
        c.translate(xx + widths[i] / 2, ty - size * 0.35)
        c.rotate(rot)
        c.scale(k * (1 + 0.04 * x.kick), k * (1 + 0.04 * x.kick))
        FX.outlined_text(c, ch, 0, size * 0.35, size, "RubikMonoOne.ttf", fill="#ffd21f", stroke="#120a18", sw=34,
                         extra_stroke="#2fe3d7", glow="#2fe3d7", shadow=True)
        c.restore()
        xx += widths[i] + 20
    if lt > 0.2:
        for k in range(2):
            FX.lightning(c, (W / 2 - 700 + k * 1400, -50), (W / 2 - 300 + k * 600, ty - 200), seed=int(t * 12) + k * 5,
                         width=5, color="#7afcff", a=0.8, branches=2)
    sub_a = clamp((lt - 0.6) / 0.3)
    FX.outlined_text(c, "~ 200 SUB SPECIAL ~", W / 2, ty + 130, 72, "Bangers.ttf", fill="#ffffff", stroke="#120a18",
                     sw=10, a=sub_a)
    if b > 2.0:
        from .timeline import bubble as bb
        bb(c, "HEY!! THAT'S MY TROPHY!!", 520, 230, (ZX + 60, 420), t - A.beat_time(2.0), 1.0, size=64, shout=True,
           fnt="Bangers.ttf", maxw=520)
    return post(bloom=0.55, threshold=0.5, impact=1.0 if lt < 0.07 else 0.0, flash=0.9 * math.exp(-lt * 6),
                chroma=6 * math.exp(-lt * 3) + 3 * x.snare, zoom_blur=0.08 * math.exp(-lt * 4),
                smear=(-200 * ease_in(clamp((b - 3.5) / 0.5), 3), 0))


# ----------------------------------------------------------------------------
# ROOFTOPS (bars 1-8)
# ----------------------------------------------------------------------------
RUN_SIZE = 350
RUN_V = 0.36 * RUN_SIZE / (0.5 * A.period)      # px/s so the feet stay planted
ROOF_T0 = bar_t(1)
LEAP_T = [A.bar_time(2), A.bar_time(3), A.bar_time(4) - A.period * 0.5]
LEAP_DUR = 0.62
ZIP_SX = 620


def xw(t):
    return RUN_V * (t - ROOF_T0)


_roofs = None


def roofs():
    global _roofs
    if _roofs is None:
        _roofs = Wd.Rooftops([xw(tl) + ZIP_SX + RUN_V * LEAP_DUR * 0.5 for tl in LEAP_T], base=840, start=-800)
    return _roofs


def s_roof_side(c, x):
    t = x.t
    R = roofs()
    scroll = xw(t)
    zoom = punch(x, 0.02)
    cam(c, zoom, ox=0, oy=0)
    Wd.sky(c, [(0, "#07051a"), (0.55, "#1f1650"), (0.85, "#4a2466"), (1, "#7a2a6a")], -100, 820)
    Wd.stars(c, t, 1.0, seed=5, n=600, h=700)
    Wd.moon(c, 1480, 190, 88)
    Wd.city(c, "far", scroll * 0.12 + 400, 760, t, scale=0.95)
    # neon billboard on the mid layer carrying the current line
    Wd.city(c, "mid", scroll * 0.35 + 900, 860, t, scale=0.95)
    for li in (0, 1):
        ln = L.lines()[li]
        a, words = ln.state(t, fade_in=0.4, fade_out=0.5)
        if a > 0:
            sx0 = 760 - (t - ln.t0) * RUN_V * 0.35
            Wd.neon_sign(c, sx0, [150, 330][li], 900, 130, words, t, color=["#ff4fb0", "#2fe3d7"][li], a=a)
    R.draw(c, scroll, t)
    # blob ahead, bouncing each beat, looking back smugly
    bxw = scroll + ZIP_SX + 700 + 60 * math.sin(t * 0.9)
    roof_b = R.roof_at(bxw) or 840
    hop = abs(math.sin(math.pi * x.u))
    br = 70
    by = roof_b - br * 0.9 - 150 * hop
    K.draw_blob(c, bxw - scroll, by, br, t, look=(-1, 0), mood="smug", arms_up=1.0, sy=1 + 0.12 * hop - 0.15 * (1 - hop),
                sx=1 - 0.08 * hop + 0.12 * (1 - hop))
    K.draw_trophy(c, bxw - scroll, by - br * 1.55, 110, t, shine=1.0)
    # Zip: run / leap
    zxw = scroll + ZIP_SX
    y_roof = R.roof_at(zxw) or 840
    pose = Z.run_pose(x.b, yaw=0.95, t=t)
    zy = y_roof - 10
    for tl in LEAP_T:
        f = (t - tl) / LEAP_DUR
        if -0.12 < f < 0:
            pose["root"] = (0.0, -Z.PELVIS_H + 0.07)
        if 0 <= f <= 1:
            y0 = (R.roof_at(xw(tl) + ZIP_SX - 5) or 840) - 10
            y1 = (R.roof_at(xw(tl + LEAP_DUR) + ZIP_SX + 5) or 840) - 10
            zy = lerp(y0, y1, f) - math.sin(f * math.pi) * 210
            pose["feet"] = {1: (0.14, -0.14), -1: (-0.2, -0.05)}
            pose["foot_ang"] = {1: -20, -1: 30}
            pose["hands"] = {1: (0.2, -0.95), -1: (-0.2, -0.85)}
            pose["elbow_hint"] = {1: (-0.5, 0.5), -1: (0.5, 0.5)}
            pose["hand_shape"] = {1: "open", -1: "open"}
            pose["expr"] = "angry"
            pose["lean"] = 10
            FX.sfx(c, "HUP!", ZIP_SX - 40, zy - 380, 70, t - tl, color="#ffffff", life=0.5, angle=-10)
        if 1 < f < 1.35:
            FX.sfx(c, "THUD", ZIP_SX + 30, zy - 20, 60, t - tl - LEAP_DUR, color="#ffd21f", life=0.45, angle=6)
    # dust at footfalls
    ph = (x.b % 0.5) / 0.5
    Wd.glow(c, ZIP_SX - 60 - ph * 80, zy - 10, 50 * (1 - ph), "#b8a8ff", 0.3 * (1 - ph))
    zdraw(c, pose, ZIP_SX, zy, RUN_SIZE, t, rim="#8a7ae8", rim_a=0.5)
    # foreground pipes whoosh by
    for j in range(3):
        X = (j * 1300 + 400 - scroll * 1.5) % (W + 1400) - 500
        c.drawRect(skia.Rect.MakeLTRB(X, 930, X + 60, H + 50), P("#07040f"))
        c.drawRect(skia.Rect.MakeLTRB(X - 30, 960, X + 90, 990), P("#07040f"))
    c.restore()
    FX.hlines(c, t, n=16, color="#c9c0ff", a=0.25, y0=300, y1=900, speed=2400, seed=2)
    first = t - ROOF_T0
    return post(bloom=0.45, threshold=0.52, flash=0.7 * math.exp(-first * 5), chroma=2 * x.snare,
                smear=(160 * math.exp(-first * 6), 0))


def s_roof_front(c, x):
    """'Blob-cam': the blob looks back at Zip sprinting straight at us."""
    t = x.t
    t0 = A.bar_time(5)
    lt = t - t0
    vp = (W / 2, 520)
    zoom = punch(x, 0.03)
    sx_, sy_ = shake(x, 5 + 6 * x.kick)
    cam(c, zoom, ox=sx_, oy=sy_)
    Wd.sky(c, [(0, "#07051a"), (0.6, "#231a58"), (1, "#6a2a6a")], -100, vp[1])
    Wd.stars(c, t, 1.0, seed=8, n=400, h=500)
    Wd.moon(c, 1560, 160, 70)
    Wd.city(c, "far", 600 + t * 20, vp[1] + 20, t, scale=0.8)
    # rooftop floor in perspective
    c.drawRect(skia.Rect.MakeLTRB(-100, vp[1], W + 100, H + 100), P(shader=lin_grad((0, vp[1]), (0, H), [(0, "#2a2150", 1), (1, "#120c26", 1)])))
    for i in range(-14, 15):
        c.drawLine(vp[0] + i * 8, vp[1], vp[0] + i * 260, H + 200, P("#3a2e6a", 0.6, stroke=2))
    speed = 1.8
    for j in range(18):
        z = (j + 1 - fract(t * speed))
        yy = vp[1] + 700 / z * 0.55
        if yy < H + 20:
            c.drawLine(-100, yy, W + 100, yy, P("#3a2e6a", clamp((yy - vp[1]) / 100) * 0.7, stroke=2 + (yy - vp[1]) / 120))
    # roadside props rushing past
    for j in range(6):
        z = ((j * 0.9 + 6) - fract(t * speed * 0.5) * 0.9 * 2 - (j // 6)) % 5.4 + 0.25
        s = 1.0 / z
        for side in (-1, 1):
            X = vp[0] + side * 900 * s
            Y = vp[1] + 420 * s
            h = 520 * s
            if side == 1 and j % 2 == 0:
                c.drawRect(skia.Rect.MakeLTRB(X, Y - h, X + 120 * s, Y), P("#1a1438"))
                c.drawCircle(X + 60 * s, Y - h - 30 * s, 50 * s, P("#1a1438"))
            else:
                c.drawLine(X, Y, X, Y - h * 1.3, P("#1a1438", 1, stroke=max(2, 16 * s)))
                on = 0.5 + 0.5 * math.sin(t * 5 + j)
                c.drawCircle(X, Y - h * 1.3, max(3, 10 * s), P("#ff3355", on))
    # goo splats thrown back at Zip on bar 7
    for k in range(4):
        ts = A.bar_time(7) + k * A.period
        age = t - ts
        if 0 < age < 0.8:
            f = age / 0.8
            gx = lerp(W / 2 + (k - 1.5) * 500, W / 2 + (k - 1.5) * 120, f)
            gy = lerp(H + 100, 520, f)
            gr = lerp(160, 20, f)
            K.draw_blob(c, gx, gy, gr, t + k, mood="panic", look=(0, -1), wob=2.0, lw=3)
    # Zip sprinting toward the camera, dodging on the goo beats
    dodge = 0.0
    for k in range(4):
        ts = A.bar_time(7) + k * A.period
        dodge += (1 if k % 2 == 0 else -1) * math.sin(math.pi * clamp((t - ts - 0.35) / 0.4)) * 0.6
    pose = Z.run_pose(x.b, yaw=0.18, stride=0.14, lift=0.24, t=t)
    pose["lean"] = 8
    ph = (x.b % 0.5) / 0.5
    pose["root"] = (0.0, pose["root"][1] - 0.035 * abs(math.sin(ph * math.pi)))
    pose["tilt"] = -12 * dodge
    pose["expr"] = "angry" if int(x.b) % 4 < 2 else "determined"
    pose["hair_v"] = (0.0, -0.18)
    pose["scarf_wind"] = (0.0, -1.0)
    size = 760
    zx = W / 2 + dodge * 260
    zy = 1120
    leap = clamp((x.b - (8 * 4 + 2.5)) / 1.5)
    if leap > 0:
        size *= 1 + 3.0 * ease_in(leap, 3)
        zy += 300 * ease_in(leap, 2)
        pose["expr"] = "rock"
        pose["hands"] = {1: (0.25, -1.0), -1: (-0.25, -1.0)}
        pose["hand_shape"] = {1: "horns", -1: "horns"}
    zdraw(c, pose, zx, zy, size, t, rim="#8a7ae8", rim_a=0.6)
    c.restore()
    for li in (2, 3):
        fly_text(c, L.lines()[li], t, vp=(W / 2, 470), size=140, pal="cool" if li == 2 else "hot")
    FX.radial_lines(c, W / 2, 520, t, n=70, inner=0.42, color="#ffffff", a=0.35)
    # goo splat on the lens at the end of bar 7
    sp = t - (A.bar_time(8) - A.period * 0.3)
    if 0 < sp < 1.2:
        k = clamp(1 - (sp - 0.6) / 0.6)
        K.draw_blob(c, 1450, 300, 260 * ease_back(clamp(sp / 0.12), 2), t, mood="shock", look=(0, 0), wob=3.0, alpha=0.9 * k)
        FX.sfx(c, "SPLAT!", 1400, 620, 150, sp, color="#ff5fa8", angle=12, life=1.0)
    return post(bloom=0.45, threshold=0.52, flash=0.8 * math.exp(-lt * 5) + 0.9 * ease_in(leap, 3), chroma=3 * x.snare,
                zoom_blur=0.05 + 0.2 * leap)


def s_roof(c, x):
    if x.t < A.bar_time(5):
        return s_roof_side(c, x)
    return s_roof_front(c, x)
