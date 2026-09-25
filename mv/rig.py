"""2D skeletal dancers drawn in a flat 'tomb painting meets cartoon' style.

Characters face +x in local space; units: ~1.0 = full height, ground at y=0, up is -y.
A pose is a dict of joint angles (degrees, screen convention: 0=right, 90=down).
Legs are solved with 2-bone IK from foot targets so steps plant on the beat.
"""
import math

import skia

from .core import (C, P, clamp, ease_io, hit, lerp, mix, poly, scale_c, smooth)

# ----------------------------------------------------------------------------
# proportions
# ----------------------------------------------------------------------------
THIGH, SHIN, ANKLE = 0.225, 0.215, 0.028
UARM, FARM, HAND = 0.148, 0.138, 0.062
TORSO = 0.265
HEAD_R = 0.09
NECK = 0.03
SHOULDER_X = 0.082
HIP_X = 0.022
PELVIS_H = 0.438

D2R = math.pi / 180


def dirv(a):
    return math.cos(a * D2R), math.sin(a * D2R)


# ----------------------------------------------------------------------------
# character definitions
# ----------------------------------------------------------------------------
CAST = {
    "nefi": dict(head="human", hair="bob", skin="skin4", hair_c="#15121c", hair_s="lapis",
                 outfit="dress", cloth="linen", belt="red", collar=("turq", "gold", "lapis", "carnelian"),
                 band="gold"),
    "khet": dict(head="human", hair="nemes", skin="skin1", hair_c="gold", hair_s="lapis",
                 outfit="kilt", cloth="linen", belt="gold", collar=("gold", "lapis", "turq", "gold"),
                 beard=True),
    "anpu": dict(head="jackal", skin="skin3", fur="#16131a", outfit="kilt", cloth="linen", belt="gold",
                 collar=("gold", "lapis", "gold", "turq"), hair="lappets", hair_c="lapis", hair_s="gold"),
    "miu": dict(head="cat", skin="skin2", fur="#1a1614", outfit="dress", cloth="#1f7a8c", belt="gold",
                collar=("gold", "turq", "gold", "red"), earring=True),
    "heru": dict(head="falcon", skin="skin1", fur="#5b3a22", outfit="kilt", cloth="linen", belt="red",
                 collar=("lapis", "gold", "turq", "gold"), hair="lappets", hair_c="lapis", hair_s="turq",
                 disk=True),
    "djehu": dict(head="ibis", skin="skin3", fur="#f4efe6", outfit="kilt", cloth="#e9d7a6", belt="lapis",
                  collar=("turq", "gold", "red", "gold"), hair="lappets", hair_c="#15121c", hair_s="lapis"),
    "sobi": dict(head="croc", skin="#4f7d3a", fur="#4f7d3a", outfit="kilt", cloth="linen", belt="gold",
                 collar=("gold", "turq", "gold", "lapis"), hair="lappets", hair_c="lapis", hair_s="gold"),
    "tut": dict(head="human", hair="nemes", skin="skin2", hair_c="gold", hair_s="#1a3fb0",
                outfit="kilt", cloth="gold", belt="lapis", collar=("lapis", "gold", "turq", "red"),
                beard=True),
    "meri": dict(head="human", hair="bob", skin="skin1", hair_c="#0e0b12", hair_s="gold",
                 outfit="dress", cloth="#b8322a", belt="gold", collar=("gold", "turq", "gold", "lapis"),
                 band="turq"),
    "mummy": dict(head="mummy", skin="#e6dcc2", fur="#e6dcc2", outfit="wrap", cloth="#e6dcc2",
                  belt=None, collar=None),
}


# ----------------------------------------------------------------------------
# poses
# ----------------------------------------------------------------------------
ARM_POSES = {
    #          front arm (upper, fore, hand)   back arm (upper, fore, hand)
    "hang": dict(fu=92, fl=86, fh=88, bu=97, bl=95, bh=94),
    "zigA": dict(fu=-4, fl=-92, fh=-4, bu=184, bl=92, bh=184),
    "zigB": dict(fu=4, fl=92, fh=4, bu=176, bl=-92, bh=176),
    "up": dict(fu=-58, fl=-96, fh=-60, bu=-122, bl=-86, bh=-120),
    "offer": dict(fu=18, fl=-2, fh=-8, bu=32, bl=4, bh=-4),
    "point": dict(fu=-30, fl=-34, fh=-36, bu=150, bl=110, bh=120),
    "hips": dict(fu=120, fl=235, fh=250, bu=60, bl=-55, bh=-70),
    "flex": dict(fu=-8, fl=-100, fh=-110, bu=188, bl=-80, bh=-90),
    "praise": dict(fu=-35, fl=-35, fh=-60, bu=-40, bl=-40, bh=-65),
    "cross": dict(fu=40, fl=-120, fh=-150, bu=60, bl=-110, bh=-140),
}


def blend(p, q, t):
    out = dict(p)
    for k, v in q.items():
        if k in p and isinstance(v, (int, float)) and isinstance(p[k], (int, float)):
            out[k] = p[k] + (v - p[k]) * t
        else:
            out[k] = v if t >= 0.5 else p.get(k, v)
    return out


def base_pose():
    p = dict(ARM_POSES["hang"])
    p.update(torso=0.0, hx=0.0, hy=0.0, htilt=0.0, py=-PELVIS_H, sx=1.0, sy=1.0,
             fa=(0.06, 0.0), ba=(-0.06, 0.0), fa_tilt=0.0, ba_tilt=0.0, blink=0.0, mouth=0.0,
             sash=0.0, vel=0.0)
    return p


# ----------------------------------------------------------------------------
# locomotion helpers
# ----------------------------------------------------------------------------
def walk(b, stride=0.17, lift=0.07, snap=0.62, speed_sign=1.0):
    """Stepping on every beat. Returns (advance, front_foot, back_foot, phase)."""
    k = math.floor(b)
    u = b - k

    def plant(j):
        return stride * j + stride * 0.5

    adv = stride * (k + u)
    s = ease_io(clamp(u / snap))
    sw = (lerp(plant(k - 1), plant(k + 1), s) - adv, -lift * math.sin(math.pi * s))
    st = (plant(k) - adv, 0.0)
    # alternate which foot swings; 'front' foot is the one drawn nearest camera
    if k % 2 == 0:
        ff, bf = sw, st
    else:
        ff, bf = st, sw
    if speed_sign < 0:
        ff, bf = (-ff[0], ff[1]), (-bf[0], bf[1])
        adv = -adv
    return adv, ff, bf, u


def step_touch(b, width=0.07, lift=0.05, snap=0.4):
    """In-place dance steps alternating forward/back."""
    k = math.floor(b)
    u = b - k
    s = ease_io(clamp(u / snap))
    pos = [width, -width * 0.2]
    a = pos[k % 2]
    c = pos[(k + 1) % 2]
    ff = (lerp(a, c, s), -lift * math.sin(math.pi * s))
    bf = (-width * 0.9, 0.0)
    return ff, bf


def groove(u, depth=0.022):
    """Pelvis dip on each beat (knees bend on the hit)."""
    return depth * (math.exp(-u * 5.0) * 1.0 - 0.15)


# ----------------------------------------------------------------------------
# choreography
# ----------------------------------------------------------------------------
def arms_seq(b, seq, snap=0.25):
    k = math.floor(b)
    u = b - k
    a = ARM_POSES[seq[(k - 1) % len(seq)]]
    c = ARM_POSES[seq[k % len(seq)]]
    return blend(a, c, hit(u, snap))


def dance(style, b, i=0, stride=0.17, energy=1.0):
    """Return (pose, advance) for a dance style at continuous beat b."""
    p = base_pose()
    k = math.floor(b)
    u = b - k
    adv = 0.0
    if style in ("walk", "egypt_walk", "moonwalk", "strut"):
        sign = -1.0 if style == "moonwalk" else 1.0
        lift = 0.09 if style == "strut" else 0.065
        adv, ff, bf, _ = walk(b, stride=stride, lift=lift, speed_sign=sign)
        p["fa"], p["ba"] = ff, bf
        p["py"] = -PELVIS_H + groove(u, 0.02 * energy)
        if style == "walk":
            sw = math.sin(b * math.pi) * 28
            p.update(fu=95 - sw, fl=80 - sw * 0.8, fh=85 - sw, bu=95 + sw, bl=95 + sw * 0.6, bh=95 + sw)
        else:
            p.update(ARM_POSES["zigA"])
            wob = math.sin(b * math.pi) * 6
            p["fu"] += wob
            p["bu"] += wob
            p["fl"] += wob * 0.6
            p["bl"] += wob * 0.6
        # head slide: forward on the beat, back on the off-beat
        p["hx"] = 0.022 * math.sin(b * math.pi * 2 - 0.4) * energy
        p["torso"] = 3 * math.sin(b * math.pi)
        p["vel"] = stride * sign
    elif style in ("toggle", "hook", "praise", "offer", "flex", "up"):
        ff, bf = step_touch(b)
        p["fa"], p["ba"] = ff, bf
        p["py"] = -PELVIS_H + groove(u, 0.03 * energy)
        if style == "toggle":
            p.update(arms_seq(b, ["zigA", "zigB"]))
        elif style == "hook":
            p.update(arms_seq(b, ["zigA", "zigB", "zigA", "up"]))
            if k % 4 == 3:
                j = math.sin(math.pi * clamp(u / 0.8)) * 0.09 * energy
                p["py"] -= j
                p["fa"] = (ff[0], ff[1] - j * 0.2)
        elif style == "praise":
            p.update(arms_seq(b, ["praise", "up", "praise", "offer"]))
        elif style == "offer":
            p.update(arms_seq(b, ["offer", "zigA"]))
        elif style == "flex":
            p.update(arms_seq(b, ["flex", "zigA", "flex", "point"]))
        elif style == "up":
            p.update(arms_seq(b, ["up", "praise"]))
        p["hx"] = 0.028 * (1 if k % 2 == 0 else -1) * hit(u, 0.2) * energy
        p["torso"] = 4 * math.sin(b * math.pi) * energy
    elif style == "idle":
        p["fa"], p["ba"] = (0.06, 0.0), (-0.05, 0.0)
        p["py"] = -PELVIS_H + 0.006 * math.sin(b * math.pi)
        p["hx"] = 0.004 * math.sin(b * 0.5)
    elif style == "freeze":
        p.update(ARM_POSES["zigA"])
        p["fa"], p["ba"] = (0.09, 0.0), (-0.08, 0.0)
        p["py"] = -PELVIS_H + 0.02
    # blink every few beats per character
    bb = (b * 0.5 + i * 1.37) % 5.0
    p["blink"] = 1.0 if bb < 0.12 else 0.0
    p["sash"] = math.sin(b * math.pi + i) * 0.5
    return p, adv


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def limb_path(a, b, wa, wb):
    """Tapered capsule from a to b with radii wa, wb."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1e-6
    nx, ny = -dy / L, dx / L
    na = math.degrees(math.atan2(dy, dx)) + 90
    path = skia.Path()
    path.moveTo(a[0] + nx * wa, a[1] + ny * wa)
    path.lineTo(b[0] + nx * wb, b[1] + ny * wb)
    path.arcTo(skia.Rect.MakeLTRB(b[0] - wb, b[1] - wb, b[0] + wb, b[1] + wb), na, -180, False)
    path.lineTo(a[0] - nx * wa, a[1] - ny * wa)
    path.arcTo(skia.Rect.MakeLTRB(a[0] - wa, a[1] - wa, a[0] + wa, a[1] + wa), na + 180, -180, False)
    path.close()
    return path


def ik2(hip, target, l1, l2, bend=1.0):
    dx, dy = target[0] - hip[0], target[1] - hip[1]
    d = math.hypot(dx, dy)
    d = clamp(d, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    a = math.atan2(dy, dx)
    cos_k = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    k = math.acos(clamp(cos_k, -1, 1))
    th = a - k * bend  # knee forward (+x) for a leg pointing down
    knee = (hip[0] + math.cos(th) * l1, hip[1] + math.sin(th) * l1)
    ang = math.atan2(target[1] - knee[1], target[0] - knee[0])
    ankle = (knee[0] + math.cos(ang) * l2, knee[1] + math.sin(ang) * l2)
    return knee, ankle


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
class Style:
    def __init__(self, mode="flat", ink="ink", ink_w=0.011, glow=None, tint=None, tint_k=0.0,
                 shade=True, alpha=1.0):
        self.mode = mode          # flat | silhouette | neon | paint
        self.ink = ink
        self.ink_w = ink_w
        self.glow = glow
        self.tint = tint
        self.tint_k = tint_k
        self.shade = shade
        self.alpha = alpha


FLAT = Style()


class Painter:
    """Draws parts honouring the style (outline pass + fill pass)."""

    def __init__(self, canvas, style, sil_color="black"):
        self.c = canvas
        self.s = style
        self.sil = sil_color

    def colr(self, c):
        s = self.s
        if s.mode == "silhouette":
            return C(self.sil)
        if s.tint is not None and s.tint_k > 0:
            return mix(c, s.tint, s.tint_k)
        return C(c)

    def part(self, paths_colors, outline=True):
        """paths_colors: list of (path, colour). Draws a unified outline then fills."""
        s = self.s
        c = self.c
        if s.mode == "neon":
            g = s.glow or colr
            for path, colr in paths_colors:
                c.drawPath(path, P(g, 0.5 * s.alpha, stroke=s.ink_w * 5, blur=s.ink_w * 2.5))
            for path, colr in paths_colors:
                c.drawPath(path, P(mix(g, "white", 0.55), s.alpha, stroke=s.ink_w * 1.6))
            for path, colr in paths_colors:
                c.drawPath(path, P(scale_c(g, 0.12), 0.96 * s.alpha))
            return
        if outline and s.mode in ("flat", "paint"):
            ink = P(s.ink, s.alpha, stroke=s.ink_w * 2)
            for path, _ in paths_colors:
                c.drawPath(path, ink)
        for path, colr in paths_colors:
            c.drawPath(path, P(self.colr(colr), s.alpha))

    def detail(self, path, colr, stroke=None, a=1.0):
        if self.s.mode in ("silhouette",):
            return
        if self.s.mode == "neon":
            self.c.drawPath(path, P(self.s.glow or colr, a * self.s.alpha, stroke=stroke or 0.006))
            return
        if stroke:
            self.c.drawPath(path, P(self.colr(colr), a * self.s.alpha, stroke=stroke))
        else:
            self.c.drawPath(path, P(self.colr(colr), a * self.s.alpha))


def draw_dancer(canvas, name, pose, x, y, size, facing=1, style=FLAT, sil_color="black", t=0.0):
    """Draw a character with feet on (x, y) screen position, height ~= size px."""
    ch = CAST[name]
    canvas.save()
    canvas.translate(x, y)
    canvas.scale(size * facing * pose.get("sx", 1.0), size * pose.get("sy", 1.0))
    pt = Painter(canvas, style, sil_color)
    _draw_body(pt, ch, pose, t)
    canvas.restore()


def _draw_body(pt, ch, p, t):
    c = pt.c
    skin = ch["skin"]
    skin_d = scale_c(C(skin), 0.78)
    wrap = ch["outfit"] == "wrap"

    pel = (0.0, p["py"])
    ta = p["torso"] * D2R
    up = (math.sin(ta), -math.cos(ta))
    right = (math.cos(ta), math.sin(ta))

    def tp(lx, ly):  # torso-local (x right, y down; negative y runs up the spine)
        return (pel[0] + right[0] * lx - up[0] * ly, pel[1] + right[1] * lx - up[1] * ly)

    neck_base = tp(0.0, -TORSO)
    sh_f = tp(SHOULDER_X, -TORSO + 0.03)
    sh_b = tp(-SHOULDER_X, -TORSO + 0.03)
    hip_f = (pel[0] + HIP_X, pel[1])
    hip_b = (pel[0] - HIP_X, pel[1])

    # ---- arms (FK)
    def arm(sh, u, l, h):
        du, dl, dh = dirv(u), dirv(l), dirv(h)
        el = (sh[0] + du[0] * UARM, sh[1] + du[1] * UARM)
        wr = (el[0] + dl[0] * FARM, el[1] + dl[1] * FARM)
        hd = (wr[0] + dh[0] * HAND, wr[1] + dh[1] * HAND)
        return el, wr, hd

    def draw_arm(sh, u, l, h, dark=False):
        el, wr, hd = arm(sh, u, l, h)
        colr = skin_d if dark else C(skin)
        hand = limb_path(wr, hd, 0.018, 0.014)
        parts = [(limb_path(sh, el, 0.03, 0.024), colr), (limb_path(el, wr, 0.024, 0.019), colr),
                 (hand, colr)]
        pt.part(parts)
        if pt.s.mode in ("flat", "paint") and not wrap:
            # armband + bracelet
            for (a, b_, f, w) in ((sh, el, 0.62, 0.03), (el, wr, 0.8, 0.024)):
                mx, my = lerp(a[0], b_[0], f), lerp(a[1], b_[1], f)
                dx, dy = b_[0] - a[0], b_[1] - a[1]
                L = math.hypot(dx, dy) or 1
                nx, ny = -dy / L * w, dx / L * w
                ax, ay = dx / L * 0.012, dy / L * 0.012
                band = poly([(mx + nx - ax, my + ny - ay), (mx + nx + ax, my + ny + ay),
                             (mx - nx + ax, my - ny + ay), (mx - nx - ax, my - ny - ay)])
                pt.detail(band, "gold")
        if wrap:
            _bandage_lines(pt, [(sh, el), (el, wr)])

    # ---- legs (IK)
    def leg(hip, foot, dark=False):
        fx, fy = foot
        ankle_t = (fx - 0.012, fy - ANKLE)
        knee, ankle = ik2(hip, ankle_t, THIGH, SHIN, bend=-1.0)
        colr = skin_d if dark else C(skin)
        lift = clamp(-fy / 0.06)
        fa = -12 * lift
        d = (math.cos(fa * D2R), math.sin(fa * D2R))
        toe = (ankle[0] + d[0] * 0.085, ankle[1] + d[1] * 0.085 + ANKLE * 0.6)
        heel = (ankle[0] - d[0] * 0.025, ankle[1] - d[1] * 0.025 + ANKLE * 0.6)
        foot_p = poly([(heel[0], heel[1] - 0.03), (ankle[0] + 0.02, ankle[1] - 0.012),
                       (toe[0], toe[1] - 0.012), (toe[0] + 0.006, toe[1] + 0.004),
                       (heel[0], heel[1] + 0.006)])
        pt.part([(limb_path(hip, knee, 0.038, 0.027), colr), (limb_path(knee, ankle, 0.027, 0.018), colr),
                 (foot_p, colr)])
        if pt.s.mode in ("flat", "paint"):
            sole = poly([(heel[0] - 0.004, heel[1] + 0.002), (toe[0] + 0.008, toe[1] + 0.001),
                         (toe[0] + 0.006, toe[1] + 0.012), (heel[0] - 0.004, heel[1] + 0.012)])
            pt.detail(sole, "#6b3f1f")
            if not wrap:
                strap = skia.Path()
                strap.moveTo(ankle[0] + 0.012, ankle[1] - 0.004)
                strap.lineTo(ankle[0] + 0.035, ankle[1] + 0.02)
                pt.detail(strap, "#6b3f1f", stroke=0.008)
        if wrap:
            _bandage_lines(pt, [(hip, knee), (knee, ankle)])
        return knee, ankle

    fa = (pel[0] + p["fa"][0], p["fa"][1])
    ba = (pel[0] + p["ba"][0], p["ba"][1])

    # draw order: back arm, back leg, front leg, torso, skirt, collar, head, front arm
    draw_arm(sh_b, p["bu"], p["bl"], p["bh"], dark=True)
    leg(hip_b, ba, dark=True)
    leg(hip_f, fa)

    # ---- torso
    torso_pts = []
    prof = [(-0.056, 0.0), (0.056, 0.0), (0.06, -0.08), (0.084, -0.18), (0.098, -0.232),
            (0.084, -0.262), (0.03, -0.272), (-0.03, -0.272), (-0.084, -0.262), (-0.098, -0.232),
            (-0.084, -0.18), (-0.06, -0.08)]
    for (lx, ly) in prof:
        torso_pts.append(tp(lx, ly))
    torso = poly(torso_pts)
    neck = limb_path(neck_base, (neck_base[0] + p["hx"] * 0.6 + 0.01, neck_base[1] - NECK - 0.02), 0.028, 0.026)
    pt.part([(neck, skin), (torso, skin)])
    if pt.s.mode == "flat" and pt.s.shade and not wrap:
        # soft shading on the back side of the torso
        sh = poly([tp(-0.058, 0.0), tp(-0.02, 0.0), tp(-0.035, -0.2), tp(-0.07, -0.25), tp(-0.086, -0.235),
                   tp(-0.078, -0.19)])
        pt.detail(sh, skin_d, a=0.55)
    if wrap:
        _bandage_torso(pt, tp)

    # ---- skirt / dress
    out = ch["outfit"]
    sash_sw = p.get("sash", 0.0) * 0.02 - p.get("vel", 0.0) * 0.08
    if out == "kilt":
        k = poly([tp(-0.066, 0.012), tp(0.064, 0.012), tp(0.135, 0.155), tp(0.02, 0.172), tp(-0.095, 0.148)])
        pt.part([(k, ch["cloth"])])
        if pt.s.mode in ("flat", "paint"):
            for j in range(5):
                f = 0.15 + j * 0.17
                pl = skia.Path()
                pl.moveTo(*tp(lerp(-0.05, 0.05, f), 0.02))
                pl.lineTo(*tp(lerp(-0.08, 0.12, f), 0.155))
                pt.detail(pl, scale_c(C(ch["cloth"]), 0.8), stroke=0.004)
            apron = poly([tp(0.01, 0.02), tp(0.07, 0.02), tp(0.135, 0.155), tp(0.05, 0.165)])
            pt.detail(apron, mix(ch["cloth"], "white", 0.25))
    elif out == "dress":
        dr = poly([tp(-0.07, -0.19), tp(0.084, -0.19), tp(0.074, -0.02), tp(0.14, 0.24),
                   tp(0.0, 0.255), tp(-0.13, 0.235), tp(-0.07, -0.02)])
        if pt.s.mode == "flat":
            pt.part([(dr, ch["cloth"])])
            for j in range(6):
                f = 0.1 + j * 0.16
                pl = skia.Path()
                pl.moveTo(*tp(lerp(-0.06, 0.07, f), -0.02))
                pl.lineTo(*tp(lerp(-0.12, 0.13, f), 0.24))
                pt.detail(pl, scale_c(C(ch["cloth"]), 0.82), stroke=0.004)
        else:
            pt.part([(dr, ch["cloth"])])
    if ch.get("belt") and out != "wrap":
        bl = poly([tp(-0.068, -0.012), tp(0.066, -0.012), tp(0.066, 0.02), tp(-0.068, 0.02)])
        pt.part([(bl, ch["belt"])], outline=pt.s.mode != "paint")
        # sash tails
        s0 = tp(0.03, 0.02)
        s1 = (s0[0] + 0.02 + sash_sw, s0[1] + 0.12)
        sash = limb_path(s0, s1, 0.012, 0.018)
        pt.part([(sash, ch["belt"])])

    # ---- collar
    if ch.get("collar"):
        cx, cy = tp(0.0, -0.262)
        cols = ch["collar"]
        rr = [0.098, 0.08, 0.063, 0.046]
        ta_deg = p["torso"]
        for j, r in enumerate(rr):
            ov = skia.Path()
            ov.addArc(skia.Rect.MakeLTRB(cx - r, cy - r * 0.62, cx + r, cy + r * 0.62), ta_deg - 5, 190)
            ov.close()
            if j == 0:
                pt.part([(ov, cols[j])])
            else:
                pt.detail(ov, cols[j])
        if pt.s.mode in ("flat", "paint"):
            for j in range(11):
                a = (ta_deg + 5 + j * 17) * D2R
                bx, by = cx + math.cos(a) * 0.096, cy + math.sin(a) * 0.096 * 0.62
                dot = skia.Path()
                dot.addCircle(bx, by, 0.007)
                pt.detail(dot, cols[3] if j % 2 else cols[1])

    # ---- head
    hc = (neck_base[0] + p["hx"] + 0.012, neck_base[1] - NECK - HEAD_R * 0.95 + p.get("hy", 0.0))
    _draw_head(pt, ch, p, hc, t)

    # ---- front arm on top
    draw_arm(sh_f, p["fu"], p["fl"], p["fh"])


def _bandage_lines(pt, segs):
    if pt.s.mode != "flat":
        return
    for a, b in segs:
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1
        nx, ny = -dy / L, dx / L
        n = 5
        for j in range(1, n):
            f = j / n
            mx, my = lerp(a[0], b[0], f), lerp(a[1], b[1], f)
            w = 0.03
            ln = skia.Path()
            ln.moveTo(mx + nx * w, my + ny * w + 0.006)
            ln.lineTo(mx - nx * w, my - ny * w - 0.004)
            pt.detail(ln, "#a89a78", stroke=0.004)


def _bandage_torso(pt, tp):
    if pt.s.mode != "flat":
        return
    for j in range(8):
        y = -0.02 - j * 0.032
        ln = skia.Path()
        ln.moveTo(*tp(-0.085, y + 0.01))
        ln.lineTo(*tp(0.085, y - 0.008))
        pt.detail(ln, "#a89a78", stroke=0.004)
    tail = skia.Path()
    tail.moveTo(*tp(-0.06, -0.15))
    tail.quadTo(*tp(-0.16, -0.1), *tp(-0.2, -0.02))
    pt.detail(tail, "#d7cba9", stroke=0.012)


# ----------------------------------------------------------------------------
# heads (unit = HEAD_R, facing right)
# ----------------------------------------------------------------------------
def _hp(pts, close=True):
    return poly(pts, close)


def _svg_path(cmds):
    """Tiny path builder: list of ('M',x,y) / ('L',x,y) / ('Q',x1,y1,x,y) / ('C',...) / ('Z',)."""
    path = skia.Path()
    for cmd in cmds:
        op = cmd[0]
        if op == "M":
            path.moveTo(cmd[1], cmd[2])
        elif op == "L":
            path.lineTo(cmd[1], cmd[2])
        elif op == "Q":
            path.quadTo(cmd[1], cmd[2], cmd[3], cmd[4])
        elif op == "C":
            path.cubicTo(cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], cmd[6])
        elif op == "Z":
            path.close()
    return path


HUMAN_HEAD = _svg_path([
    ("M", -0.95, 0.05), ("C", -0.98, -0.72, -0.4, -1.04, 0.12, -1.02),
    ("C", 0.55, -1.0, 0.8, -0.72, 0.83, -0.38), ("L", 0.86, -0.18), ("Q", 1.02, 0.02, 1.13, 0.2),
    ("Q", 1.12, 0.29, 0.96, 0.3), ("L", 0.99, 0.41), ("Q", 0.94, 0.47, 0.9, 0.49), ("L", 0.96, 0.57),
    ("Q", 0.95, 0.76, 0.78, 0.86), ("C", 0.6, 0.98, 0.3, 0.98, 0.1, 0.9), ("C", -0.35, 0.75, -0.9, 0.6, -0.95, 0.05),
    ("Z",)])

BOB_WIG = _svg_path([
    ("M", 0.74, -0.52), ("C", 0.64, -1.2, -0.55, -1.32, -0.98, -0.62), ("C", -1.2, -0.1, -1.12, 0.6, -0.95, 1.12),
    ("L", 0.08, 1.14), ("L", 0.12, 0.38), ("C", 0.18, -0.05, 0.36, -0.3, 0.74, -0.34), ("Z",)])

NEMES = _svg_path([
    ("M", 0.78, -0.5), ("C", 0.7, -1.3, -0.62, -1.4, -1.0, -0.72), ("L", -1.42, 0.55),
    ("Q", -1.32, 1.25, -0.75, 1.3), ("L", 0.02, 0.98), ("L", 0.18, 0.12), ("Q", 0.34, -0.42, 0.78, -0.5), ("Z",)])

LAPPETS = _svg_path([
    ("M", 0.6, -0.62), ("C", 0.4, -1.25, -0.7, -1.3, -0.98, -0.6), ("L", -1.25, 0.9), ("L", -0.55, 1.35),
    ("L", -0.15, 0.95), ("L", -0.05, 0.1), ("Q", 0.1, -0.5, 0.6, -0.62), ("Z",)])


def _eye(pt, cx, cy, s=1.0, blink=0.0, iris="#120b08", kohl=True, color_white="#fbf5e6"):
    h = 0.14 * s * (1 - blink * 0.9)
    e = _svg_path([("M", cx - 0.2 * s, cy), ("Q", cx, cy - h * 1.3, cx + 0.2 * s, cy - 0.01),
                   ("Q", cx, cy + h, cx - 0.2 * s, cy), ("Z",)])
    pt.detail(e, color_white)
    if blink < 0.5:
        ir = skia.Path()
        ir.addCircle(cx + 0.05 * s, cy - 0.012, 0.068 * s)
        pt.detail(ir, iris)
        hl = skia.Path()
        hl.addCircle(cx + 0.075 * s, cy - 0.04, 0.02 * s)
        pt.detail(hl, "white", a=0.9)
    if kohl:
        pt.detail(e, "#0c0806", stroke=0.055 * s)
        wing = _svg_path([("M", cx - 0.19 * s, cy), ("Q", cx - 0.35 * s, cy + 0.02, cx - 0.52 * s, cy - 0.04)])
        pt.detail(wing, "#0c0806", stroke=0.06 * s)
    brow = _svg_path([("M", cx - 0.42 * s, cy - 0.2 * s), ("Q", cx, cy - 0.38 * s, cx + 0.24 * s, cy - 0.22 * s)])
    pt.detail(brow, "#0c0806", stroke=0.065 * s)


def _stripes(pt, clip, c1, c2, n=9, angle=0.0, x0=-1.6, x1=1.6, y0=-1.6, y1=1.6):
    if pt.s.mode != "flat" and pt.s.mode != "paint":
        return
    c = pt.c
    c.save()
    c.clipPath(clip, doAntiAlias=True)
    c.rotate(angle)
    step = (y1 - y0) / n
    for j in range(n):
        if j % 2:
            r = skia.Rect.MakeLTRB(x0, y0 + j * step, x1, y0 + (j + 1) * step)
            c.drawRect(r, P(pt.colr(c2), pt.s.alpha))
    c.restore()


def _draw_head(pt, ch, p, hc, t):
    c = pt.c
    c.save()
    c.translate(*hc)
    c.rotate(p.get("htilt", 0.0))
    c.scale(HEAD_R, HEAD_R)
    kind = ch["head"]
    s = pt.s
    iw = s.ink_w / HEAD_R  # ink width in head units
    saved = s.ink_w
    s.ink_w = iw
    try:
        if kind == "human":
            _head_human(pt, ch, p)
        elif kind == "jackal":
            _head_jackal(pt, ch, p)
        elif kind == "cat":
            _head_cat(pt, ch, p)
        elif kind == "falcon":
            _head_falcon(pt, ch, p)
        elif kind == "ibis":
            _head_ibis(pt, ch, p)
        elif kind == "croc":
            _head_croc(pt, ch, p)
        elif kind == "mummy":
            _head_mummy(pt, ch, p, t)
    finally:
        s.ink_w = saved
    c.restore()


def _back_hair(pt, ch):
    hair = ch.get("hair")
    if hair == "nemes":
        pt.part([(NEMES, ch["hair_c"])])
        _stripes(pt, NEMES, ch["hair_c"], ch["hair_s"], n=11, angle=-8)
        lap = _svg_path([("M", 0.05, 0.6), ("L", 0.42, 0.62), ("L", 0.62, 2.1), ("L", 0.2, 2.15), ("Z",)])
        pt.part([(lap, ch["hair_c"])])
        _stripes(pt, lap, ch["hair_c"], ch["hair_s"], n=9, y0=0.5, y1=2.2)
    elif hair == "lappets":
        pt.part([(LAPPETS, ch["hair_c"])])
        _stripes(pt, LAPPETS, ch["hair_c"], ch["hair_s"], n=12, angle=-4)


def _head_human(pt, ch, p):
    hair = ch.get("hair")
    if hair == "nemes":
        _back_hair(pt, ch)
    pt.part([(HUMAN_HEAD, ch["skin"])])
    if pt.s.mode == "flat" and pt.s.shade:
        cheek = skia.Path()
        cheek.addOval(skia.Rect.MakeLTRB(0.28, 0.12, 0.62, 0.4))
        pt.detail(cheek, "#ff6b6b", a=0.18)
    if hair == "bob":
        pt.part([(BOB_WIG, ch["hair_c"])])
        _stripes(pt, BOB_WIG, ch["hair_c"], ch["hair_s"], n=14, angle=0, y0=-1.4, y1=1.2)
        band = _svg_path([("M", 0.76, -0.5), ("C", 0.3, -0.62, -0.5, -0.66, -1.05, -0.4)])
        pt.detail(band, ch.get("band", "gold"), stroke=0.14)
        lotus = _svg_path([("M", 0.62, -0.55), ("Q", 0.66, -0.9, 0.82, -1.0), ("Q", 0.9, -0.72, 0.74, -0.52), ("Z",)])
        pt.detail(lotus, "turq")
    elif hair == "nemes":
        band = _svg_path([("M", 0.78, -0.5), ("C", 0.45, -0.66, -0.2, -0.72, -0.75, -0.62)])
        pt.detail(band, "gold", stroke=0.13)
        # uraeus
        ur = _svg_path([("M", 0.7, -0.55), ("Q", 0.9, -0.8, 0.8, -1.0), ("Q", 0.95, -1.02, 0.98, -0.9),
                        ("Q", 0.95, -0.72, 0.82, -0.5), ("Z",)])
        pt.part([(ur, "gold")])
    _eye(pt, 0.45, -0.1, 1.0, p.get("blink", 0))
    if ch.get("beard"):
        bd = _svg_path([("M", 0.62, 0.84), ("L", 0.82, 0.8), ("L", 0.9, 1.55), ("L", 0.72, 1.58), ("Z",)])
        pt.part([(bd, "#2a1a10")])
        for j in range(4):
            ln = _svg_path([("M", 0.66 + j * 0.02, 0.98 + j * 0.15), ("L", 0.86 + j * 0.01, 0.95 + j * 0.15)])
            pt.detail(ln, "gold", stroke=0.035)
    mouth = _svg_path([("M", 0.93, 0.49), ("Q", 0.84, 0.54, 0.78, 0.5)])
    pt.detail(mouth, "#5a2318", stroke=0.05)


def _head_jackal(pt, ch, p):
    _back_hair(pt, ch)
    fur = ch["fur"]
    ear_b = _hp([(-0.55, -0.55), (-0.5, -2.05), (-0.05, -0.75)])
    ear_f = _hp([(-0.25, -0.62), (-0.05, -2.2), (0.3, -0.7)])
    head = _svg_path([("M", -0.85, 0.3), ("C", -0.95, -0.55, -0.4, -0.95, 0.15, -0.85),
                      ("C", 0.5, -0.8, 0.7, -0.55, 0.95, -0.35), ("L", 1.82, -0.02), ("Q", 1.9, 0.1, 1.78, 0.18),
                      ("L", 1.0, 0.36), ("Q", 0.6, 0.55, 0.3, 0.62), ("C", -0.1, 0.9, -0.7, 0.8, -0.85, 0.3), ("Z",)])
    pt.part([(ear_b, fur), (head, fur), (ear_f, fur)])
    inner = _hp([(-0.13, -0.8), (-0.03, -1.9), (0.18, -0.78)])
    pt.detail(inner, "gold", a=0.9)
    nose = skia.Path()
    nose.addCircle(1.8, 0.06, 0.1)
    pt.detail(nose, "#000000")
    jaw = _svg_path([("M", 1.7, 0.2), ("Q", 1.2, 0.35, 0.75, 0.3)])
    pt.detail(jaw, "#3a2b24", stroke=0.05)
    _eye(pt, 0.42, -0.3, 0.8, p.get("blink", 0), iris="#e8b21a", color_white="#f6d36b")
    stripe = _svg_path([("M", 0.2, -0.26), ("Q", 0.4, 0.05, 0.1, 0.35)])
    pt.detail(stripe, "gold", stroke=0.07)


def _head_cat(pt, ch, p):
    fur = ch["fur"]
    ear_b = _hp([(-0.65, -0.45), (-0.55, -1.45), (-0.05, -0.78)])
    ear_f = _hp([(-0.2, -0.72), (0.15, -1.55), (0.5, -0.6)])
    head = _svg_path([("M", -0.9, 0.1), ("C", -0.95, -0.7, -0.3, -1.0, 0.2, -0.95), ("C", 0.65, -0.9, 0.9, -0.55, 0.92, -0.2),
                      ("Q", 1.2, -0.02, 1.2, 0.2), ("Q", 1.15, 0.45, 0.85, 0.52), ("Q", 0.6, 0.85, 0.2, 0.85),
                      ("C", -0.4, 0.8, -0.88, 0.6, -0.9, 0.1), ("Z",)])
    pt.part([(ear_b, fur), (head, fur), (ear_f, fur)])
    inner = _hp([(0.0, -0.75), (0.17, -1.35), (0.38, -0.66)])
    pt.detail(inner, "#c46a7a", a=0.85)
    nose = _hp([(1.08, 0.02), (1.22, 0.06), (1.16, 0.16)])
    pt.detail(nose, "#e07a8a")
    _eye(pt, 0.52, -0.18, 0.85, p.get("blink", 0), iris="#0b0b0b", color_white="#7bd88f", kohl=True)
    for j in range(3):
        wh = _svg_path([("M", 0.95, 0.25 + j * 0.08), ("Q", 1.3, 0.2 + j * 0.12, 1.6, 0.18 + j * 0.2)])
        pt.detail(wh, "#e8e0d0", stroke=0.025)
    if ch.get("earring"):
        ring = skia.Path()
        ring.addCircle(-0.1, 0.55, 0.16)
        pt.detail(ring, "gold", stroke=0.07)


def _head_falcon(pt, ch, p):
    _back_hair(pt, ch)
    fur = ch["fur"]
    head = _svg_path([("M", -0.85, 0.2), ("C", -0.9, -0.6, -0.35, -0.98, 0.15, -0.95), ("C", 0.6, -0.9, 0.9, -0.55, 0.9, -0.2),
                      ("L", 1.02, -0.12), ("Q", 1.42, 0.0, 1.4, 0.45), ("Q", 1.3, 0.62, 1.2, 0.55), ("Q", 1.18, 0.3, 0.95, 0.3),
                      ("Q", 0.6, 0.7, 0.2, 0.8), ("C", -0.4, 0.8, -0.8, 0.6, -0.85, 0.2), ("Z",)])
    pt.part([(head, fur)])
    throat = _svg_path([("M", 0.62, 0.02), ("Q", 0.9, 0.08, 0.98, 0.3), ("Q", 0.62, 0.72, 0.12, 0.8),
                        ("Q", 0.5, 0.45, 0.62, 0.02), ("Z",)])
    pt.detail(throat, "#f3ead8")
    beak = _svg_path([("M", 1.0, -0.1), ("Q", 1.42, 0.0, 1.4, 0.45), ("Q", 1.3, 0.62, 1.2, 0.55), ("Q", 1.18, 0.3, 0.95, 0.3), ("Z",)])
    pt.detail(beak, "#f2b632")
    pt.detail(beak, "#241a10", stroke=0.05)
    tear = _svg_path([("M", 0.36, -0.12), ("C", 0.5, 0.1, 0.46, 0.35, 0.38, 0.62), ("L", 0.2, 0.66),
                      ("C", 0.3, 0.4, 0.3, 0.12, 0.2, -0.08), ("Z",)])
    pt.detail(tear, "#1a120b")
    _eye(pt, 0.5, -0.25, 0.8, p.get("blink", 0), iris="#0b0b0b", color_white="#ffd24a", kohl=True)
    if ch.get("disk") and pt.s.mode == "flat":
        disk = skia.Path()
        disk.addCircle(-0.1, -1.55, 0.62)
        pt.part([(disk, "#e0402a")])
        ring = skia.Path()
        ring.addCircle(-0.1, -1.55, 0.46)
        pt.detail(ring, "#ff8a3a", a=0.8)


def _head_ibis(pt, ch, p):
    _back_hair(pt, ch)
    fur = ch["fur"]
    head = _svg_path([("M", -0.75, 0.2), ("C", -0.85, -0.6, -0.3, -0.9, 0.15, -0.85), ("C", 0.55, -0.8, 0.8, -0.5, 0.82, -0.18),
                      ("Q", 0.7, 0.4, 0.2, 0.7), ("C", -0.3, 0.8, -0.72, 0.6, -0.75, 0.2), ("Z",)])
    beak = _svg_path([("M", 0.72, -0.28), ("C", 1.6, -0.2, 2.3, 0.5, 2.55, 1.35), ("C", 2.15, 0.75, 1.5, 0.2, 0.72, 0.12), ("Z",)])
    pt.part([(beak, "#2b2724"), (head, "#1d1a22")])
    sheen = _svg_path([("M", -0.4, -0.72), ("Q", 0.2, -0.95, 0.6, -0.6)])
    pt.detail(sheen, "#4a5aa8", stroke=0.12, a=0.8)
    ridge = _svg_path([("M", 0.9, -0.16), ("C", 1.6, -0.05, 2.1, 0.5, 2.45, 1.2)])
    pt.detail(ridge, "#6d655e", stroke=0.05)
    ring = skia.Path()
    ring.addCircle(0.45, -0.2, 0.2)
    pt.detail(ring, "#f6f0e2")
    _eye(pt, 0.45, -0.2, 0.62, p.get("blink", 0), iris="#b01e1e", color_white="#fff2c8", kohl=True)


def _head_croc(pt, ch, p):
    _back_hair(pt, ch)
    fur = ch["fur"]
    head = _svg_path([("M", -0.85, 0.2), ("C", -0.9, -0.55, -0.35, -0.9, 0.2, -0.8), ("Q", 0.55, -0.72, 0.7, -0.45),
                      ("L", 2.05, -0.2), ("Q", 2.2, -0.05, 2.05, 0.1), ("L", 0.85, 0.35), ("Q", 0.4, 0.7, 0.1, 0.75),
                      ("C", -0.4, 0.8, -0.8, 0.6, -0.85, 0.2), ("Z",)])
    pt.part([(head, fur)])
    brow = _svg_path([("M", 0.15, -0.72), ("Q", 0.42, -1.05, 0.7, -0.62), ("Z",)])
    pt.part([(brow, fur)])
    mouth = _svg_path([("M", 2.0, 0.02), ("L", 0.7, 0.2)])
    pt.detail(mouth, "#1f2d15", stroke=0.05)
    for j in range(7):
        x = 0.85 + j * 0.17
        tooth = _hp([(x, 0.18 - (x - 0.7) * 0.12), (x + 0.06, 0.28 - (x - 0.7) * 0.12), (x + 0.1, 0.16 - (x - 0.7) * 0.12)])
        pt.detail(tooth, "#fffbe8")
    for j in range(4):
        sc = skia.Path()
        sc.addCircle(0.9 + j * 0.28, -0.35 + j * 0.035, 0.05)
        pt.detail(sc, "#2f5222")
    _eye(pt, 0.42, -0.5, 0.65, p.get("blink", 0), iris="#0b0b0b", color_white="#e6d23a", kohl=True)


def _head_mummy(pt, ch, p, t):
    pt.part([(HUMAN_HEAD, ch["skin"])])
    if pt.s.mode == "flat":
        c = pt.c
        c.save()
        c.clipPath(HUMAN_HEAD, doAntiAlias=True)
        for j in range(10):
            y = -1.1 + j * 0.24
            ln = _svg_path([("M", -1.2, y + 0.1 * (j % 2)), ("Q", 0.0, y - 0.12, 1.3, y + 0.05)])
            pt.detail(ln, "#a89a78", stroke=0.05)
        c.restore()
        slot = _svg_path([("M", 0.25, -0.2), ("Q", 0.55, -0.3, 0.8, -0.12), ("Q", 0.55, 0.02, 0.25, -0.2), ("Z",)])
        pt.detail(slot, "#1b120a")
        glow = 0.75 + 0.25 * math.sin(t * 9)
        g = skia.Path()
        g.addCircle(0.55, -0.16, 0.09)
        pt.c.drawPath(g, P("#7dffcf", glow, blur=0.12))
        pt.detail(g, "#e9fff6")
