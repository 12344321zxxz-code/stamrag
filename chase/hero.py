"""ZIP - the (original) punk guitarist. A 2.5D cel-shaded puppet.

Local space: 1.0 ~= body height, origin on the ground under the pelvis, up is -y, faces +x.
`yaw` 0 = facing camera, 1 = right profile.  Mirror the whole puppet to face left.
Legs and arms are 2-bone IK; knees always bend toward the facing direction (or outward when
seen from the front), elbows bend back/outward.
"""
import math

import skia

from .core import (C, P, clamp, ease_io, ease_out, hit, lerp, lin_grad, mix, path_from, poly, rad_grad, scale_c,
                   smooth)

INK = "#170f1f"
SKIN, SKIN_SH = "#f7c8a2", "#dc9878"
HAIR, HAIR_SH, HAIR_HI = "#2fe3d7", "#1592a6", "#caffF9"
JACKET, JACKET_HI, JACKET_SH = "#25222f", "#46425c", "#15131b"
TEE, TEE_SH = "#f4f2ec", "#cfcad9"
BOLT_Y = "#ffd21f"
JEANS, JEANS_SH, JEANS_HI = "#30508f", "#223a6d", "#5579c0"
SHOE, SHOE_SH, SOLE = "#ec3441", "#a91c2a", "#f8f4ea"
SCARF, SCARF_SH = "#ff3d3d", "#b91c2c"
GLOVE = "#1b1822"
IRIS1, IRIS2 = "#ffd04a", "#ff6a00"

THIGH, SHIN, ANKLE = 0.215, 0.205, 0.035
UARM, FARM = 0.135, 0.125
TORSO, NECK = 0.25, 0.03
HEAD_R = 0.115
SHOULDER_W, HIP_W = 0.09, 0.05
PELVIS_H = 0.43
D2R = math.pi / 180

EXPR = {
    "neutral": dict(open=0.9, brow=0.0, brow_tilt=0.0, mouth="flat", mo=0.0, look=(0.0, 0.0)),
    "smug": dict(open=0.62, brow=-0.15, brow_tilt=0.35, mouth="smirk", mo=0.0, look=(0.1, 0.0)),
    "deadpan": dict(open=0.5, brow=0.0, brow_tilt=0.0, mouth="flat", mo=0.0, look=(0.0, 0.0)),
    "shock": dict(open=1.25, brow=0.6, brow_tilt=-0.2, mouth="o", mo=0.8, look=(0.0, 0.0), small_pupil=True),
    "angry": dict(open=0.85, brow=-0.7, brow_tilt=0.0, mouth="yell", mo=0.7, look=(0.0, 0.0)),
    "determined": dict(open=0.8, brow=-0.55, brow_tilt=0.0, mouth="grit", mo=0.35, look=(0.25, 0.0)),
    "grin": dict(open=0.0, brow=0.25, brow_tilt=0.0, mouth="grin", mo=0.6, look=(0.0, 0.0), happy=True),
    "disgust": dict(open=0.55, brow=0.0, brow_tilt=0.8, mouth="wavy", mo=0.2, look=(0.0, 0.0)),
    "wink": dict(open=0.9, brow=0.1, brow_tilt=0.3, mouth="grin", mo=0.45, look=(0.0, 0.0), wink=True),
    "rock": dict(open=0.35, brow=-0.45, brow_tilt=0.0, mouth="yell", mo=0.9, look=(0.0, 0.0)),
    "focus": dict(open=0.45, brow=-0.35, brow_tilt=0.0, mouth="grit", mo=0.15, look=(0.2, 0.25)),
    "soft": dict(open=0.75, brow=0.3, brow_tilt=-0.2, mouth="smile", mo=0.1, look=(0.0, 0.0)),
}


def base_pose(yaw=0.6):
    return dict(
        yaw=yaw, head_yaw=None, lean=4.0, tilt=0.0, root=(0.0, -PELVIS_H), head_tilt=0.0, head_nod=0.0,
        feet={1: (0.06, 0.0), -1: (-0.05, 0.0)}, foot_ang={1: 0.0, -1: 0.0},
        hands={1: None, -1: None}, hand_shape={1: "fist", -1: "fist"}, elbow_hint={1: None, -1: None},
        arm_front={1: False, -1: False},
        guitar="back", strum=0.0, fret=0.5, gtr_ang=-24.0, gtr_glow=0.0,
        expr="neutral", talk=0.0, blink=0.0, hair_v=(0.0, 0.0), hair_bounce=0.0, speed=0.0,
        scarf_wind=(-1.0, 0.0), aura=0.0, eye_glow=0.0, shade_eyes=0.0, sx=1.0, sy=1.0, t=0.0,
    )


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def capsule(a, b, wa, wb):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1e-6
    nx, ny = -dy / L, dx / L
    na = math.degrees(math.atan2(dy, dx)) + 90
    p = skia.Path()
    p.moveTo(a[0] + nx * wa, a[1] + ny * wa)
    p.lineTo(b[0] + nx * wb, b[1] + ny * wb)
    p.arcTo(skia.Rect.MakeLTRB(b[0] - wb, b[1] - wb, b[0] + wb, b[1] + wb), na, -180, False)
    p.lineTo(a[0] - nx * wa, a[1] - ny * wa)
    p.arcTo(skia.Rect.MakeLTRB(a[0] - wa, a[1] - wa, a[0] + wa, a[1] + wa), na + 180, -180, False)
    p.close()
    return p


def ik(hip, target, l1, l2, hint, k=1.0):
    """2-bone IK. hint: 2D vector the joint should bulge toward. k in [0,1] foreshortens the bend."""
    dx, dy = target[0] - hip[0], target[1] - hip[1]
    D = math.hypot(dx, dy)
    D = clamp(D, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    ux, uy = (dx, dy) if math.hypot(dx, dy) > 1e-6 else (0.0, 1.0)
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    a = (l1 * l1 - l2 * l2 + D * D) / (2 * D)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    nx, ny = -uy, ux
    if nx * hint[0] + ny * hint[1] < 0:
        nx, ny = -nx, -ny
    mid = (hip[0] + ux * a + nx * h * k, hip[1] + uy * a + ny * h * k)
    end = (hip[0] + ux * D, hip[1] + uy * D)
    return mid, end


class Ink:
    """Outline-then-fill painter with cel shading and optional rim light."""

    def __init__(self, c, lw, light=(-0.6, -0.8), rim=None, rim_a=0.0, sil=None, alpha=1.0, tint=None, tint_k=0.0):
        self.c = c
        self.lw = lw
        self.light = light
        self.rim = rim
        self.rim_a = rim_a
        self.sil = sil
        self.alpha = alpha
        self.tint = tint
        self.tint_k = tint_k

    def col(self, c):
        if self.sil is not None:
            return C(self.sil)
        if self.tint is not None and self.tint_k > 0:
            return mix(c, self.tint, self.tint_k)
        return C(c)

    def group(self, parts, outline=True):
        """parts: list of (path, fill, shadow_or_None)."""
        c = self.c
        if outline and self.sil is None:
            ink = P(INK, self.alpha, stroke=self.lw * 2)
            for path, _, _ in parts:
                c.drawPath(path, ink)
        for path, fill, shade in parts:
            c.drawPath(path, P(self.col(fill), self.alpha))
            if self.sil is None and shade is not None:
                self.shade(path, shade)

    def shade(self, path, shade, off=0.022):
        c = self.c
        c.save()
        c.clipPath(path, skia.ClipOp.kIntersect, True)
        lx, ly = self.light
        sh = skia.Path(path)
        sh.offset(-lx * off, -ly * off)
        inv = skia.Path()
        b = path.getBounds()
        inv.addRect(skia.Rect.MakeLTRB(b.left() - 1, b.top() - 1, b.right() + 1, b.bottom() + 1))
        c.clipPath(sh, skia.ClipOp.kDifference, True)
        c.drawPath(inv, P(self.col(shade), self.alpha))
        if self.rim is not None and self.rim_a > 0:
            rp = skia.Path(path)
            rp.offset(lx * off * 0.6, ly * off * 0.6)
            c.restore()
            c.save()
            c.clipPath(path, skia.ClipOp.kIntersect, True)
            c.clipPath(rp, skia.ClipOp.kDifference, True)
            c.drawPath(inv, P(self.rim, self.rim_a * self.alpha))
        c.restore()

    def detail(self, path, colr, stroke=None, a=1.0):
        if self.sil is not None:
            return
        if stroke:
            self.c.drawPath(path, P(self.col(colr), a * self.alpha, stroke=stroke))
        else:
            self.c.drawPath(path, P(self.col(colr), a * self.alpha))


# ----------------------------------------------------------------------------
# hair: spikes in head-local 3D (lat, dep, y) -> tip offsets
# ----------------------------------------------------------------------------
# (root lat, root dep, root y, tip dlat, tip ddep, tip dy, width, layer)  layer 0 back, 1 top, 2 bangs
HAIR_SPIKES = [
    # back spikes (behind the skull)
    (-0.55, -0.55, -0.35, -0.35, -0.55, 0.25, 0.42, 0), (0.55, -0.55, -0.35, 0.35, -0.55, 0.25, 0.42, 0),
    (0.0, -0.8, -0.2, 0.0, -0.75, 0.35, 0.5, 0), (-0.3, -0.75, 0.25, -0.15, -0.45, 0.5, 0.38, 0),
    (0.3, -0.75, 0.25, 0.15, -0.45, 0.5, 0.38, 0),
    # the big quiff: rises and sweeps back
    (-0.35, 0.35, -0.85, -0.2, -0.55, -1.05, 0.46, 1), (0.35, 0.35, -0.85, 0.2, -0.55, -1.05, 0.46, 1),
    (0.0, 0.55, -0.8, 0.0, -0.35, -1.35, 0.55, 1), (-0.6, 0.0, -0.75, -0.35, -0.7, -0.75, 0.4, 1),
    (0.6, 0.0, -0.75, 0.35, -0.7, -0.75, 0.4, 1), (0.0, 0.2, -0.95, 0.0, -0.95, -0.95, 0.5, 1),
    # bangs over the forehead (stop above the eyes)
    (-0.42, 0.78, -0.62, -0.12, 0.1, 0.42, 0.32, 2), (0.36, 0.8, -0.62, 0.14, 0.1, 0.4, 0.3, 2),
    (-0.02, 0.9, -0.66, 0.06, 0.1, 0.36, 0.26, 2),
    # sideburns
    (-0.92, 0.22, -0.25, -0.04, 0.06, 0.5, 0.22, 2), (0.92, 0.22, -0.25, 0.04, 0.06, 0.5, 0.22, 2),
]

# hairline loop across the front of the head (lat, dep, y); the cap covers everything above it
HAIRLINE = [(-0.97, 0.15, 0.2), (-0.85, 0.45, -0.28), (-0.5, 0.75, -0.5), (0.0, 0.88, -0.56), (0.5, 0.75, -0.5),
            (0.85, 0.45, -0.28), (0.97, 0.15, 0.2)]


class Head:
    def __init__(self, yaw):
        th = clamp(yaw) * 78 * D2R
        self.c, self.s = math.cos(th), math.sin(th)

    def proj(self, lat, dep, y):
        return (lat * self.c + dep * self.s, y)

    def facing(self, lat, dep):
        L = math.hypot(lat, dep) or 1.0
        return (dep * self.c - lat * self.s) / L


def _spike(hd, sp, hv, bounce):
    lat, dep, y, dl, dd, dy, w, _ = sp
    r = hd.proj(lat, dep, y)
    tip = hd.proj(lat + dl, dep + dd, y + dy)
    L = math.hypot(tip[0] - r[0], tip[1] - r[1]) or 1e-6
    tip = (tip[0] + hv[0] * L, tip[1] + hv[1] * L + bounce * L * 0.35)
    ux, uy = (tip[0] - r[0]) / L, (tip[1] - r[1]) / L
    nx, ny = -uy, ux
    bw = w * 0.5
    a = (r[0] + nx * bw, r[1] + ny * bw)
    b = (r[0] - nx * bw, r[1] - ny * bw)
    bend = 0.18 * L
    m1 = ((a[0] + tip[0]) / 2 + nx * bend * 0.3, (a[1] + tip[1]) / 2 + ny * bend * 0.3)
    m2 = ((b[0] + tip[0]) / 2 + nx * bend * 0.1, (b[1] + tip[1]) / 2 + ny * bend * 0.1)
    p = skia.Path()
    p.moveTo(*a)
    p.quadTo(m1[0], m1[1], *tip)
    p.quadTo(m2[0], m2[1], *b)
    p.close()
    return p, r, tip


def draw_head(ink, pose, t):
    """Head-local drawing; the canvas is already translated to the skull centre and scaled by HEAD_R."""
    c = ink.c
    hyaw = pose["head_yaw"] if pose["head_yaw"] is not None else pose["yaw"]
    hd = Head(hyaw)
    ex = dict(EXPR.get(pose["expr"], EXPR["neutral"]))
    hv = pose["hair_v"]
    bounce = pose["hair_bounce"]
    lw = ink.lw
    ink.lw = lw / HEAD_R
    try:
        # back hair
        back = []
        for sp in HAIR_SPIKES:
            if sp[7] == 0:
                pth, _, _ = _spike(hd, sp, hv, bounce)
                back.append((pth, HAIR, HAIR_SH))
        ink.group(back)
        # ears (near side only)
        for sd in (-1, 1):
            f = hd.facing(0.95 * sd, 0.0)
            if f > 0.2:
                ec = hd.proj(0.93 * sd, -0.05, 0.12)
                ear = skia.Path()
                ear.addOval(skia.Rect.MakeLTRB(ec[0] - 0.16 * f, ec[1] - 0.22, ec[0] + 0.16 * f, ec[1] + 0.22))
                ink.group([(ear, SKIN, SKIN_SH)])
        # skull + jaw silhouette (convex hull of circle samples + projected jaw points)
        pts = [(math.cos(a) * 1.0, math.sin(a) * 1.0) for a in [i * math.pi / 18 for i in range(36)]]
        jaw = [(-0.82, 0.25, 0.42), (0.82, 0.25, 0.42), (-0.5, 0.6, 0.82), (0.5, 0.6, 0.82), (0.0, 0.66, 1.02),
               (-0.9, 0.3, 0.12), (0.9, 0.3, 0.12), (0.0, 1.1, 0.26)]
        pts += [hd.proj(*j) for j in jaw]
        hull = _hull(pts)
        face = path_from(hull, close=True, smooth_k=0.35)
        ink.group([(face, SKIN, None)])
        # soft cheek shading on the far side
        if ink.sil is None:
            c.save()
            c.clipPath(face, skia.ClipOp.kIntersect, True)
            fx = -hd.s * 0.9 - 0.25
            c.drawCircle(fx - 0.9, 0.2, 1.3, P(SKIN_SH, 0.35 * ink.alpha))
            c.restore()
            # blush
            for sd in (-1, 1):
                f = hd.facing(0.55 * sd, 0.75)
                if f > 0.1:
                    bc = hd.proj(0.52 * sd, 0.75, 0.42)
                    c.drawOval(skia.Rect.MakeLTRB(bc[0] - 0.14 * f, bc[1] - 0.06, bc[0] + 0.14 * f, bc[1] + 0.06),
                               P("#ff7a8a", 0.35 * ink.alpha))
        # eyes
        for sd in (-1, 1):
            _eye(ink, hd, sd, ex, pose, t)
        # nose
        if ink.sil is None:
            nt = hd.proj(0.0, 1.02, 0.4)
            nb = hd.proj(0.0, 0.9, 0.47)
            np_ = skia.Path()
            np_.moveTo(nt[0], nt[1] - 0.12)
            np_.lineTo(nt[0] + 0.05 * hd.s + 0.02, nt[1] + 0.02)
            np_.lineTo(nb[0] - 0.05, nb[1])
            ink.detail(np_, SKIN_SH, stroke=0.05)
        _mouth(ink, hd, ex, pose)
        # top hair cap: the skull above the (projected) hairline, plus the back of the head in profile
        hl = [hd.proj(*p) for p in HAIRLINE if hd.facing(p[0], p[1]) > -0.02]
        above = [(-3.0, hl[0][1] + 0.3)] + hl + [(3.0, hl[-1][1] + 0.3), (3.0, -3.0), (-3.0, -3.0)]
        skull = skia.Path()
        skull.addCircle(0, 0, 1.05)
        capp = skia.Op(skull, path_from(above, close=True, smooth_k=0.3), skia.PathOp.kIntersect_PathOp)
        if hd.s > 0.2:
            ear_x = hd.proj(-0.93, -0.05, 0.0)[0]
            back = skia.Path()
            back.addRect(skia.Rect.MakeLTRB(-3.0, -3.0, ear_x - 0.12, 0.62))
            capp = skia.Op(capp, skia.Op(skull, back, skia.PathOp.kIntersect_PathOp), skia.PathOp.kUnion_PathOp)
        top = [(capp, HAIR, HAIR_SH)]
        for sp in HAIR_SPIKES:
            if sp[7] == 1:
                pth, _, _ = _spike(hd, sp, hv, bounce)
                top.append((pth, HAIR, HAIR_SH))
        ink.group(top)
        # highlight streaks on the quiff
        if ink.sil is None:
            for sp in HAIR_SPIKES:
                if sp[7] == 1 and abs(sp[0]) < 0.4:
                    _, r, tip = _spike(hd, sp, hv, bounce)
                    hl = skia.Path()
                    hl.moveTo(lerp(r[0], tip[0], 0.25) - 0.05, lerp(r[1], tip[1], 0.25))
                    hl.quadTo(lerp(r[0], tip[0], 0.5) - 0.12, lerp(r[1], tip[1], 0.5), lerp(r[0], tip[0], 0.75) - 0.06,
                              lerp(r[1], tip[1], 0.75))
                    ink.detail(hl, HAIR_HI, stroke=0.07, a=0.8)
        # eyebrows (drawn through the bangs, anime style)
        for sd in (-1, 1):
            _brow(ink, hd, sd, ex)
        bangs = []
        for sp in HAIR_SPIKES:
            if sp[7] == 2:
                f = hd.facing(sp[0], sp[1])
                if f > -0.2 or abs(sp[0]) < 0.5:
                    pth, _, _ = _spike(hd, sp, hv, bounce * 0.4)
                    bangs.append((pth, HAIR, HAIR_SH))
        ink.group(bangs)
        # lightning hair clip on the near side
        if ink.sil is None:
            cc = hd.proj(-0.62, 0.45, -0.62)
            if hd.facing(-0.62, 0.45) > 0:
                bolt = poly([(cc[0] - 0.12, cc[1] - 0.2), (cc[0] + 0.1, cc[1] - 0.2), (cc[0] - 0.02, cc[1] - 0.02),
                             (cc[0] + 0.12, cc[1] - 0.02), (cc[0] - 0.14, cc[1] + 0.26), (cc[0] - 0.04, cc[1] + 0.04),
                             (cc[0] - 0.16, cc[1] + 0.04)])
                ink.group([(bolt, BOLT_Y, None)])
        if pose.get("shade_eyes", 0) > 0 and ink.sil is None:
            sh = skia.Path()
            sh.addRect(skia.Rect.MakeLTRB(-1.3, -0.35, 1.5, 0.36))
            c.save()
            c.clipPath(face, skia.ClipOp.kIntersect, True)
            c.drawPath(sh, P("#0b0612", 0.8 * pose["shade_eyes"] * ink.alpha))
            c.restore()
            if pose.get("eye_glow", 0) > 0:
                for sd in (-1, 1):
                    f = hd.facing(0.4 * sd, 0.86)
                    if f > 0.05:
                        e = hd.proj(0.4 * sd, 0.86, 0.16)
                        c.drawCircle(e[0], e[1], 0.16, P("#fff4a0", pose["eye_glow"] * 0.6, blur=0.12))
                        c.drawCircle(e[0], e[1], 0.05, P("#ffffff", pose["eye_glow"]))
    finally:
        ink.lw = lw


def _hull(pts):
    pts = sorted(set((round(x, 5), round(y, 5)) for x, y in pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _eye(ink, hd, sd, ex, pose, t):
    lat, dep, y = 0.4 * sd, 0.86, 0.16
    f = hd.facing(lat, dep)
    if f < 0.08:
        return
    cx, cy = hd.proj(lat, dep, y)
    w = 0.31 * (0.35 + 0.65 * f)
    op = ex["open"] * (1 - pose.get("blink", 0.0))
    if ex.get("wink") and sd == 1:
        op = 0.0
    c = ink.c
    if ex.get("happy") or op < 0.08:
        arc = skia.Path()
        arc.moveTo(cx - w, cy + 0.02)
        arc.quadTo(cx, cy - 0.22 if ex.get("happy") else cy + 0.06, cx + w, cy + 0.02)
        ink.detail(arc, INK, stroke=0.085)
        return
    h = 0.4 * op
    tilt = -0.06 * sd * 0  # outer corners slightly down
    eye = skia.Path()
    eye.moveTo(cx - w, cy + 0.03)
    eye.cubicTo(cx - w * 0.6, cy - h * 1.05, cx + w * 0.6, cy - h * 1.1, cx + w, cy - 0.02 + tilt)
    eye.cubicTo(cx + w * 0.7, cy + h * 0.75, cx - w * 0.7, cy + h * 0.8, cx - w, cy + 0.03)
    eye.close()
    ink.detail(eye, "#fffdf7")
    lx, ly = ex.get("look", (0, 0))
    ix, iy = cx + (lx + 0.25 * hd.s) * w * 0.5, cy - h * 0.05 + ly * h * 0.4
    c.save()
    c.clipPath(eye, skia.ClipOp.kIntersect, True)
    ir = 0.2 * (0.55 + 0.45 * f) if not ex.get("small_pupil") else 0.13
    irp = skia.Path()
    irp.addOval(skia.Rect.MakeLTRB(ix - ir * 0.8, iy - ir, ix + ir * 0.8, iy + ir))
    if ink.sil is None:
        c.drawPath(irp, P(shader=lin_grad((ix, iy - ir), (ix, iy + ir), [(0, IRIS2, ink.alpha), (0.6, IRIS1, ink.alpha),
                                                                          (1, "#fff0a0", ink.alpha)])))
        c.drawOval(skia.Rect.MakeLTRB(ix - ir * 0.38, iy - ir * 0.5, ix + ir * 0.38, iy + ir * 0.5), P("#2a0f08", ink.alpha))
        c.drawCircle(ix + ir * 0.35, iy - ir * 0.45, ir * 0.3, P("#ffffff", ink.alpha))
        c.drawCircle(ix - ir * 0.3, iy + ir * 0.45, ir * 0.14, P("#ffffff", 0.8 * ink.alpha))
        if pose.get("eye_glow", 0) > 0:
            c.drawCircle(ix, iy, ir * 1.4, P("#fff2a0", pose["eye_glow"] * 0.6 * ink.alpha))
    c.restore()
    # thick upper lash line
    lash = skia.Path()
    lash.moveTo(cx - w * 1.08, cy + 0.02)
    lash.cubicTo(cx - w * 0.6, cy - h * 1.1, cx + w * 0.6, cy - h * 1.15, cx + w * 1.12, cy - 0.04)
    ink.detail(lash, INK, stroke=0.1)
    wing = skia.Path()
    ox = cx + w * 1.1 * (1 if sd == -1 else -1) * (1 if hd.s < 0.2 else 1)
    wing.moveTo(cx + w * 1.05, cy - 0.03)
    wing.lineTo(cx + w * 1.25, cy - 0.12)
    ink.detail(wing, INK, stroke=0.07)
    low = skia.Path()
    low.moveTo(cx - w * 0.6, cy + h * 0.72)
    low.quadTo(cx, cy + h * 0.85, cx + w * 0.55, cy + h * 0.62)
    ink.detail(low, INK, stroke=0.035, a=0.7)


def _brow(ink, hd, sd, ex):
    lat, dep = 0.42 * sd, 0.82
    f = hd.facing(lat, dep)
    if f < 0.05:
        return
    cx, cy = hd.proj(lat, dep, -0.22 - 0.12 * ex["brow"])
    w = 0.28 * (0.4 + 0.6 * f)
    # inner end (toward nose) goes down when angry (brow<0)
    inner_dx = -sd * w
    tilt = ex.get("brow_tilt", 0.0) * (0.12 if sd == 1 else -0.04)
    inner_y = cy - ex["brow"] * 0.0 + (-ex["brow"]) * 0.1 * (1 if ex["brow"] < 0 else -0.5) - tilt
    outer_y = cy - 0.06 + tilt
    b = skia.Path()
    b.moveTo(cx + inner_dx, inner_y)
    b.quadTo(cx, min(inner_y, outer_y) - 0.08, cx - inner_dx, outer_y)
    ink.detail(b, "#0f5d6a", stroke=0.11)


def _mouth(ink, hd, ex, pose):
    f = hd.facing(0.0, 1.0)
    cx, cy = hd.proj(0.0, 0.92, 0.68)
    cx += 0.08 * hd.s
    w = 0.26 * (0.45 + 0.55 * max(0.0, f))
    mo = max(ex["mo"], pose.get("talk", 0.0))
    shape = ex["mouth"]
    if pose.get("talk", 0.0) > 0.05 and shape in ("flat", "smirk", "smile"):
        shape = "talk"
    c = ink.c
    if shape in ("flat",):
        m = skia.Path()
        m.moveTo(cx - w * 0.6, cy)
        m.lineTo(cx + w * 0.5, cy)
        ink.detail(m, INK, stroke=0.055)
    elif shape in ("smirk", "smile"):
        m = skia.Path()
        m.moveTo(cx - w * 0.6, cy + (0.0 if shape == "smirk" else -0.03))
        m.quadTo(cx, cy + 0.1, cx + w * 0.7, cy - (0.1 if shape == "smirk" else 0.03))
        ink.detail(m, INK, stroke=0.055)
        if shape == "smirk":
            fang = poly([(cx + w * 0.15, cy + 0.045), (cx + w * 0.32, cy + 0.03), (cx + w * 0.22, cy + 0.14)])
            ink.detail(fang, "#ffffff")
    elif shape == "wavy":
        m = skia.Path()
        m.moveTo(cx - w * 0.6, cy)
        m.cubicTo(cx - w * 0.2, cy - 0.08, cx + w * 0.1, cy + 0.08, cx + w * 0.6, cy - 0.02)
        ink.detail(m, INK, stroke=0.05)
    else:
        oh = 0.08 + 0.26 * clamp(mo)
        ww = w * (0.55 if shape == "o" else 1.0 if shape in ("yell", "grin") else 0.8)
        m = skia.Path()
        if shape == "grin":
            m.moveTo(cx - ww, cy - 0.04)
            m.quadTo(cx, cy + 0.02, cx + ww, cy - 0.06)
            m.quadTo(cx + ww * 0.3, cy + oh * 1.2, cx - ww * 0.2, cy + oh * 0.9)
            m.close()
        elif shape == "grit":
            m.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(cx - ww, cy - oh * 0.35, cx + ww, cy + oh * 0.45), 0.05, 0.05))
        else:
            m.addOval(skia.Rect.MakeLTRB(cx - ww, cy - oh * 0.5, cx + ww, cy + oh * 0.7))
        ink.detail(m, "#4a1020")
        ink.detail(m, INK, stroke=0.05)
        c.save()
        c.clipPath(m, skia.ClipOp.kIntersect, True)
        if shape in ("grin", "grit", "yell"):
            c.drawRect(skia.Rect.MakeLTRB(cx - ww * 1.2, cy - oh, cx + ww * 1.2, cy - oh * 0.2 + 0.06 + (0.08 if shape == "grit" else 0)),
                       P("#ffffff", ink.alpha))
        if shape in ("yell", "o", "talk"):
            c.drawOval(skia.Rect.MakeLTRB(cx - ww * 0.6, cy + oh * 0.2, cx + ww * 0.6, cy + oh * 0.9), P("#ff6f7d", ink.alpha))
        c.restore()


# ----------------------------------------------------------------------------
# guitar (original "Bolt" design) in guitar-local coords: origin at bridge, +x toward headstock
# ----------------------------------------------------------------------------
GTR_BODY = [(-0.22, -0.02), (-0.14, -0.16), (-0.04, -0.1), (0.02, -0.19), (0.1, -0.07), (0.13, -0.035),
            (0.13, 0.04), (0.06, 0.06), (0.12, 0.17), (0.0, 0.1), (-0.08, 0.2), (-0.12, 0.08), (-0.24, 0.12),
            (-0.17, 0.02)]


def draw_guitar(ink, glow=0.0, strum=0.0, t=0.0, string_vib=0.0):
    c = ink.c
    body = path_from(GTR_BODY, close=True, smooth_k=0.12)
    neck = poly([(0.1, -0.022), (0.53, -0.017), (0.53, 0.017), (0.1, 0.022)])
    head = poly([(0.52, -0.03), (0.6, -0.05), (0.7, -0.012), (0.66, 0.0), (0.7, 0.014), (0.6, 0.05), (0.52, 0.03)])
    if glow > 0:
        c.drawPath(body, P("#c46bff", 0.5 * glow * ink.alpha, stroke=0.05, blur=0.03))
    ink.group([(body, "#7b2cff", "#4e16b0"), (neck, "#e9c27f", "#b88a4a"), (head, "#7b2cff", "#4e16b0")])
    if ink.sil is not None:
        return
    # binding + pickguard + pickups
    c.drawPath(body, P(ink.col(BOLT_Y), ink.alpha, stroke=0.012))
    guard = path_from([(-0.12, -0.06), (0.0, -0.08), (0.08, -0.03), (0.07, 0.05), (-0.04, 0.06), (-0.13, 0.03)],
                      close=True, smooth_k=0.3)
    c.drawPath(guard, P(ink.col("#141018"), ink.alpha))
    board = poly([(0.1, -0.016), (0.53, -0.012), (0.53, 0.012), (0.1, 0.016)])
    c.drawPath(board, P(ink.col("#3a2317"), ink.alpha))
    for i in range(14):
        fx = 0.53 - (0.43 * (1 - 2 ** (-(i + 1) / 6.0)) / (1 - 2 ** (-14 / 6.0)))
        c.drawLine(fx, -0.015, fx, 0.015, P("#d8dde6", ink.alpha, stroke=0.003))
        if i in (2, 4, 6, 8, 11):
            c.drawCircle(fx + 0.012, 0.0, 0.0045, P("#fff8e0", ink.alpha))
    for px in (-0.03, 0.045):
        c.drawRoundRect(skia.Rect.MakeLTRB(px - 0.018, -0.035, px + 0.018, 0.035), 0.008, 0.008, P("#0b0a10", ink.alpha))
        for k in range(6):
            c.drawCircle(px, -0.028 + k * 0.0112, 0.0035, P("#c9ced9", ink.alpha))
    c.drawRect(skia.Rect.MakeLTRB(-0.085, -0.04, -0.07, 0.04), P("#c9ced9", ink.alpha))
    for k in range(3):
        c.drawCircle(-0.15 + k * 0.035, 0.07 + k * 0.02, 0.012, P("#e8e8f0", ink.alpha))
    for k in range(3):
        c.drawCircle(0.585 + k * 0.035, -0.045 + k * 0.006, 0.009, P("#d8dde6", ink.alpha))
        c.drawCircle(0.585 + k * 0.035, 0.045 - k * 0.006, 0.009, P("#d8dde6", ink.alpha))
    # strings (vibrate when played)
    for k in range(6):
        y0 = -0.014 + k * 0.0056
        amp = string_vib * 0.004 * (1 + k * 0.2)
        sp = skia.Path()
        sp.moveTo(-0.078, y0 * 2.2)
        n = 8
        for j in range(1, n + 1):
            xx = lerp(-0.078, 0.53, j / n)
            yy = lerp(y0 * 2.2, y0, j / n) + math.sin(j / n * math.pi) * amp * math.sin(t * 90 + k * 1.7 + j)
            sp.lineTo(xx, yy)
        col_ = mix("#f5f0e6", "#fff6a0", glow)
        c.drawPath(sp, P(ink.col(col_), ink.alpha, stroke=0.0022 + 0.002 * glow))
        if glow > 0.05:
            c.drawPath(sp, P("#fff39a", 0.5 * glow * ink.alpha, stroke=0.01, blur=0.008))


# ----------------------------------------------------------------------------
# the full puppet
# ----------------------------------------------------------------------------
def draw_zip(canvas, pose, x, y, size, mirror=False, light=(-0.6, -0.8), rim=None, rim_a=0.0, sil=None, alpha=1.0,
             tint=None, tint_k=0.0, lw=0.0115):
    canvas.save()
    canvas.translate(x, y)
    canvas.scale(size * (-1 if mirror else 1) * pose.get("sx", 1.0), size * pose.get("sy", 1.0))
    ink = Ink(canvas, lw, (light[0] * (-1 if mirror else 1), light[1]), rim, rim_a, sil, alpha, tint, tint_k)
    _draw_body(ink, pose)
    canvas.restore()


def joints(pose):
    """Compute all joint positions (local space) for a pose."""
    yaw = clamp(pose["yaw"])
    th = yaw * 78 * D2R
    cth, sth = math.cos(th), math.sin(th)
    px, py = pose["root"]
    ang = pose["lean"] * sth + pose["tilt"]
    a = ang * D2R
    u = (math.sin(a), -math.cos(a))
    r = (math.cos(a), math.sin(a))
    pel = (px, py)
    neck = (px + u[0] * TORSO, py + u[1] * TORSO)
    J = dict(yaw=yaw, cth=cth, sth=sth, u=u, r=r, pel=pel, neck=neck, ang=ang)
    for sd in (-1, 1):
        J[("sh", sd)] = (neck[0] + r[0] * sd * SHOULDER_W * cth - u[0] * 0.035 + sth * 0.0,
                         neck[1] + r[1] * sd * SHOULDER_W * cth - u[1] * 0.035)
        J[("hip", sd)] = (px + r[0] * sd * HIP_W * cth, py + r[1] * sd * HIP_W * cth)
        J[("near", sd)] = -sd * sth
    # legs
    for sd in (-1, 1):
        fx, fy = pose["feet"][sd]
        hip = J[("hip", sd)]
        ank_t = (fx, fy - ANKLE)
        dirx = lerp(0.45 * sd, 1.0, sth)
        knee, ank = ik(hip, ank_t, THIGH, SHIN, (1.0 if dirx >= 0 else -1.0, 0.0), k=min(1.0, abs(dirx) + 0.15))
        J[("knee", sd)], J[("ankle", sd)] = knee, ank
    # arms
    for sd in (-1, 1):
        sh = J[("sh", sd)]
        tgt = pose["hands"][sd]
        if tgt is None:
            tgt = (sh[0] + sd * 0.05 * cth - 0.02 * sth, sh[1] + UARM + FARM - 0.03)
        hint = pose["elbow_hint"][sd]
        if hint is None:
            hint = (lerp(sd * 0.9, -1.0, sth), 0.55)
        el, wr = ik(sh, tgt, UARM, FARM, hint, k=1.0)
        J[("elbow", sd)], J[("wrist", sd)] = el, wr
    # head
    hc = (neck[0] + u[0] * (NECK + HEAD_R * 0.92) + pose.get("head_nod", 0.0) * sth * 0.4,
          neck[1] + u[1] * (NECK + HEAD_R * 0.92) + pose.get("head_nod", 0.0))
    J["head"] = hc
    return J


def guitar_xf(pose, J):
    """(origin, angle_deg, sx, sy, behind) for the guitar in local space."""
    mode = pose["guitar"]
    u, r, pel, sth, cth = J["u"], J["r"], J["pel"], J["sth"], J["cth"]
    if mode == "play":
        o = (pel[0] - r[0] * 0.03 * cth + u[0] * 0.07 + sth * 0.075, pel[1] - r[1] * 0.03 + u[1] * 0.07)
        return o, pose["gtr_ang"] + J["ang"] * 0.5, lerp(1.0, 0.8, sth), lerp(1.0, 0.62, sth), False
    if mode == "back":
        o = (pel[0] - sth * 0.07 + u[0] * 0.05, pel[1] + u[1] * 0.05)
        return o, -118 + J["ang"], 0.95, 0.7, True
    if mode == "ride":
        return (pel[0] - 0.02, 0.012), 180.0 if pose.get("ride_flip") else 0.0, 1.35, 0.55, True
    if mode == "raise":
        o = (pel[0] + u[0] * 0.25 + 0.05, pel[1] + u[1] * 0.25)
        return o, pose["gtr_ang"], 1.0, 0.9, False
    return None


def _draw_body(ink, pose):
    c = ink.c
    J = joints(pose)
    sth, cth, u, r = J["sth"], J["cth"], J["u"], J["r"]
    t = pose.get("t", 0.0)
    far = 1 if sth >= 0 else -1          # character's left side is away from camera when facing right
    near = -far
    if sth < 0.12:
        far, near = 1, -1
    # aura
    if pose.get("aura", 0) > 0 and ink.sil is None:
        pel = J["pel"]
        c.drawCircle(pel[0], pel[1] - 0.2, 0.55, P("#7afcff", 0.25 * pose["aura"], blur=0.12))
    _scarf_tails(ink, pose, J, t)
    gx = guitar_xf(pose, J) if pose["guitar"] != "none" else None
    if gx and gx[4]:
        _place_guitar(ink, pose, gx, t)
    play = pose["guitar"] == "play"
    # far arm (behind torso) unless flagged in front
    if not (play or pose["arm_front"][far]):
        _arm(ink, J, far, pose, dark=True)
    # legs
    _leg(ink, J, far, pose, dark=True)
    _leg(ink, J, near, pose)
    _torso(ink, J, pose)
    _scarf_wrap(ink, J)
    # head
    hc = J["head"]
    c.save()
    c.translate(*hc)
    c.rotate(pose.get("head_tilt", 0.0) + J["ang"] * 0.3)
    c.scale(HEAD_R, HEAD_R)
    draw_head(ink, pose, t)
    c.restore()
    if play:
        # fret arm (character's left = +1) under the neck, then guitar, then fret fingers, then strum arm
        _arm(ink, J, 1, pose, dark=(far == 1), hand=False)
        _place_guitar(ink, pose, gx, t)
        _hand(ink, J, 1, pose)
        _arm(ink, J, -1, pose)
    else:
        if pose["arm_front"][far]:
            _arm(ink, J, far, pose, dark=True)
        if gx and not gx[4]:
            _place_guitar(ink, pose, gx, t)
        _arm(ink, J, near, pose)


def _place_guitar(ink, pose, gx, t):
    c = ink.c
    o, ang, sx, sy, _ = gx
    c.save()
    c.translate(*o)
    c.rotate(ang)
    c.scale(sx * 1.05, sy * 1.05)
    lw = ink.lw
    ink.lw = lw / max(sx, 0.3)
    draw_guitar(ink, pose.get("gtr_glow", 0.0), pose.get("strum", 0.0), t, string_vib=pose.get("string_vib", 0.0))
    ink.lw = lw
    c.restore()


def guitar_point(pose, J, gx_, gy_):
    """Local-space position of a guitar-local point (used for hand targets)."""
    o, ang, sx, sy, _ = guitar_xf(pose, J)
    a = ang * D2R
    x, y = gx_ * sx * 1.05, gy_ * sy * 1.05
    return (o[0] + x * math.cos(a) - y * math.sin(a), o[1] + x * math.sin(a) + y * math.cos(a))


def _leg(ink, J, sd, pose, dark=False):
    hip, knee, ank = J[("hip", sd)], J[("knee", sd)], J[("ankle", sd)]
    sth = J["sth"]
    jeans = scale_c(C(JEANS), 0.82) if dark else JEANS
    th = capsule(hip, knee, 0.052, 0.043)
    sh = capsule(knee, ank, 0.043, 0.034)
    # shoe
    fa = pose["foot_ang"][sd] * D2R
    L = 0.105 * lerp(0.5, 1.0, sth)
    fwd = (math.cos(fa), math.sin(fa))
    heel = (ank[0] - fwd[0] * 0.03 * lerp(0.6, 1, sth), ank[1] + ANKLE * 0.7)
    toe = (ank[0] + fwd[0] * L, ank[1] + fwd[1] * L + ANKLE * 0.75)
    w = lerp(0.052, 0.04, sth)
    shoe = path_from([(heel[0] - 0.005, heel[1] - 0.05), (ank[0] + fwd[0] * 0.02, ank[1] - 0.045),
                      (toe[0] - fwd[0] * 0.03, toe[1] - 0.038), (toe[0] + 0.004, toe[1] - 0.012),
                      (toe[0], toe[1] + 0.008), (heel[0], heel[1] + 0.008)], close=True, smooth_k=0.25)
    if sth < 0.3:
        shoe = skia.Path()
        cx = ank[0] + fwd[0] * 0.02
        shoe.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(cx - w, ank[1] - 0.045, cx + w, ank[1] + ANKLE * 0.75 + 0.008),
                                            0.03, 0.03))
    ink.group([(th, jeans, JEANS_SH), (sh, jeans, JEANS_SH)])
    ink.group([(shoe, SHOE, SHOE_SH)])
    if ink.sil is None:
        b = shoe.getBounds()
        ink.c.save()
        ink.c.clipPath(shoe, skia.ClipOp.kIntersect, True)
        ink.c.drawRect(skia.Rect.MakeLTRB(b.left() - 0.01, b.bottom() - 0.022, b.right() + 0.01, b.bottom() + 0.01), P(ink.col(SOLE), ink.alpha))
        ink.c.restore()
        # knee rip on the near leg
        if not dark:
            k = knee
            rip = skia.Path()
            rip.addOval(skia.Rect.MakeLTRB(k[0] - 0.024, k[1] - 0.012, k[0] + 0.024, k[1] + 0.014))
            ink.detail(rip, "#d6def0")
            for j in range(3):
                ln = skia.Path()
                ln.moveTo(k[0] - 0.022, k[1] - 0.006 + j * 0.008)
                ln.lineTo(k[0] + 0.022, k[1] - 0.004 + j * 0.008)
                ink.detail(ln, "#ffffff", stroke=0.003)


def _arm(ink, J, sd, pose, dark=False, hand=True):
    sh, el, wr = J[("sh", sd)], J[("elbow", sd)], J[("wrist", sd)]
    jk = scale_c(C(JACKET), 0.8) if dark else JACKET
    sk = scale_c(C(SKIN), 0.9) if dark else SKIN
    up = capsule(sh, el, 0.038, 0.033)
    fo = capsule(el, wr, 0.03, 0.026)
    ink.group([(fo, sk, SKIN_SH), (up, jk, JACKET_SH)])
    if ink.sil is None:
        # rolled cuff at the elbow
        dx, dy = el[0] - sh[0], el[1] - sh[1]
        L = math.hypot(dx, dy) or 1
        cuff = capsule((el[0] - dx / L * 0.02, el[1] - dy / L * 0.02), (el[0] + dx / L * 0.012, el[1] + dy / L * 0.012),
                       0.041, 0.04)
        ink.group([(cuff, JACKET_HI, None)])
    if hand:
        _hand(ink, J, sd, pose)


def _hand(ink, J, sd, pose):
    el, wr = J[("elbow", sd)], J[("wrist", sd)]
    dx, dy = wr[0] - el[0], wr[1] - el[1]
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    shape = pose["hand_shape"][sd]
    c = ink.c
    c.save()
    c.translate(*wr)
    c.rotate(math.degrees(math.atan2(uy, ux)))
    # hand local: +x along the forearm
    if shape == "open":
        palm = path_from([(0.0, -0.03), (0.05, -0.034), (0.085, -0.02), (0.088, 0.02), (0.05, 0.032), (0.0, 0.03)],
                         close=True, smooth_k=0.3)
        thumb = capsule((0.02, -0.028), (0.05, -0.058), 0.012, 0.01)
        ink.group([(thumb, SKIN, None), (palm, GLOVE, None)])
        for k in range(4):
            f = capsule((0.07, -0.022 + k * 0.015), (0.11 - abs(k - 1.5) * 0.008, -0.024 + k * 0.016), 0.0075, 0.0065)
            ink.group([(f, SKIN, None)], outline=True)
    elif shape == "point":
        fist = path_from([(0.0, -0.03), (0.06, -0.032), (0.07, 0.0), (0.06, 0.03), (0.0, 0.03)], close=True, smooth_k=0.3)
        idx = capsule((0.055, -0.02), (0.115, -0.024), 0.009, 0.008)
        ink.group([(idx, SKIN, None), (fist, GLOVE, None)])
    elif shape == "horns":
        fist = path_from([(0.0, -0.03), (0.06, -0.032), (0.07, 0.0), (0.06, 0.03), (0.0, 0.03)], close=True, smooth_k=0.3)
        f1 = capsule((0.055, -0.022), (0.11, -0.03), 0.009, 0.008)
        f2 = capsule((0.055, 0.022), (0.11, 0.03), 0.009, 0.008)
        ink.group([(f1, SKIN, None), (f2, SKIN, None), (fist, GLOVE, None)])
    elif shape == "pick":
        fist = path_from([(0.0, -0.03), (0.055, -0.032), (0.068, 0.0), (0.055, 0.03), (0.0, 0.03)], close=True,
                         smooth_k=0.3)
        pick = poly([(0.062, -0.012), (0.09, 0.0), (0.062, 0.012)])
        ink.group([(pick, "#ffd21f", None), (fist, GLOVE, None)])
    elif shape == "fret":
        palm = path_from([(0.0, -0.028), (0.05, -0.03), (0.06, 0.0), (0.05, 0.028), (0.0, 0.028)], close=True, smooth_k=0.3)
        ink.group([(palm, GLOVE, None)])
        for k in range(4):
            f = capsule((0.045, -0.02 + k * 0.013), (0.07, -0.028 + k * 0.016), 0.0075, 0.0065)
            ink.group([(f, SKIN, None)])
    else:  # fist
        fist = path_from([(0.0, -0.032), (0.06, -0.036), (0.078, 0.0), (0.06, 0.034), (0.0, 0.032)], close=True,
                         smooth_k=0.3)
        ink.group([(fist, GLOVE, None)])
        if ink.sil is None:
            for k in range(3):
                ln = skia.Path()
                ln.moveTo(0.06, -0.018 + k * 0.016)
                ln.lineTo(0.074, -0.018 + k * 0.016)
                ink.detail(ln, SKIN, stroke=0.008)
    c.restore()


def _torso(ink, J, pose):
    pel, neck, u, r, sth, cth = J["pel"], J["neck"], J["u"], J["r"], J["sth"], J["cth"]

    def at(h, lat, dep):
        # h: 0 at pelvis .. 1 at neck; lat: lateral half-width multiplier; dep: forward depth
        bx, by = pel[0] + u[0] * TORSO * h, pel[1] + u[1] * TORSO * h
        x = lat * cth + dep * sth
        return (bx + r[0] * x, by + r[1] * x)

    rows = [(-0.12, 0.085, 0.06), (0.0, 0.08, 0.058), (0.35, 0.078, 0.06), (0.72, 0.095, 0.068), (0.9, 0.098, 0.06),
            (1.0, 0.07, 0.045)]
    left, right = [], []
    for h, wf, dd in rows:
        # silhouette extremes of an ellipse (wf lateral, dd depth) rotated by yaw
        hw = math.hypot(wf * cth, dd * sth)
        off = 0.0
        right.append(at(h, 0, 0))
        right[-1] = (right[-1][0] + r[0] * (hw + off), right[-1][1] + r[1] * (hw + off))
        left.append(at(h, 0, 0))
        left[-1] = (left[-1][0] - r[0] * (hw - off), left[-1][1] - r[1] * (hw - off))
    outline = right + list(reversed(left))
    jacket = path_from(outline, close=True, smooth_k=0.25)
    ink.group([(jacket, JACKET, JACKET_SH)])
    if ink.sil is not None:
        return
    c = ink.c
    # tee strip (visible front opening), shifts toward the facing side
    fcx = 0.058 * sth
    sw = 0.045 * cth + 0.006
    tee = path_from([at(0.02, -0.045, 0.058 if sth > 0 else 0), at(0.02, 0.045, 0.058 if sth > 0 else 0),
                     at(0.97, 0.03, 0.05), at(0.97, -0.03, 0.05)], close=True)
    tp = []
    for h in (0.03, 0.5, 0.97):
        b = at(h, 0, 0)
        cx = b[0] + r[0] * fcx
        cy = b[1] + r[1] * fcx
        tp.append(((cx - r[0] * sw, cy - r[1] * sw), (cx + r[0] * sw, cy + r[1] * sw)))
    teep = poly([tp[0][0], tp[1][0], tp[2][0], tp[2][1], tp[1][1], tp[0][1]])
    c.save()
    c.clipPath(jacket, skia.ClipOp.kIntersect, True)
    ink.group([(teep, TEE, TEE_SH)], outline=False)
    # lightning print
    b = at(0.55, 0, 0)
    bx, by = b[0] + r[0] * fcx, b[1] + r[1] * fcx
    s = 0.03 * max(0.3, cth)
    bolt = poly([(bx - s * 0.2, by - s * 1.4), (bx + s * 0.9, by - s * 1.4), (bx + s * 0.1, by - s * 0.2),
                 (bx + s * 0.8, by - s * 0.2), (bx - s * 0.7, by + s * 1.6), (bx - s * 0.1, by + s * 0.2),
                 (bx - s * 0.8, by + s * 0.2)])
    ink.detail(bolt, BOLT_Y)
    # lapels
    for sd in (-1, 1):
        a0, a1 = tp[2][0 if sd < 0 else 1], tp[0][0 if sd < 0 else 1]
        lap = poly([a0, (a0[0] + sd * 0.03 * cth, a0[1] + 0.02), (lerp(a0[0], a1[0], 0.55) + sd * 0.012, lerp(a0[1], a1[1], 0.55)),
                    (lerp(a0[0], a1[0], 0.6), lerp(a0[1], a1[1], 0.6))])
        ink.detail(lap, JACKET_HI)
    # zipper line + hem
    hem = poly([at(-0.12, -0.2, 0), at(-0.12, 0.2, 0), at(-0.06, 0.2, 0), at(-0.06, -0.2, 0)])
    ink.detail(hem, "#1a1822")
    c.restore()
    # studded belt peeking under the jacket
    for k in range(5):
        p = at(-0.05, -0.06 + k * 0.03, 0.05)
        if abs(-0.06 + k * 0.03) * cth < 0.075:
            c.drawCircle(p[0], p[1], 0.006, P(ink.col("#d9dde6"), ink.alpha))
    # jacket highlight edge
    hl = skia.Path()
    p0, p1 = at(0.88, 0, 0), at(0.25, 0, 0)
    hw0 = math.hypot(0.098 * cth, 0.06 * sth)
    hl.moveTo(p0[0] - r[0] * hw0 * 0.85, p0[1] - r[1] * hw0 * 0.85)
    hl.lineTo(p1[0] - r[0] * hw0 * 0.75, p1[1] - r[1] * hw0 * 0.75)
    ink.detail(hl, JACKET_HI, stroke=0.012, a=0.8)


def _scarf_wrap(ink, J):
    neck = J["neck"]
    u, r = J["u"], J["r"]
    w = math.hypot(0.075 * J["cth"], 0.055 * J["sth"])
    band = path_from([(neck[0] - r[0] * w, neck[1] - r[1] * w + 0.008), (neck[0] + r[0] * w, neck[1] + r[1] * w + 0.008),
                      (neck[0] + r[0] * w * 0.85 + u[0] * 0.05, neck[1] + r[1] * w * 0.85 + u[1] * 0.05 - 0.004),
                      (neck[0] - r[0] * w * 0.85 + u[0] * 0.05, neck[1] - r[1] * w * 0.85 + u[1] * 0.05 - 0.004)],
                     close=True, smooth_k=0.35)
    ink.group([(band, SCARF, SCARF_SH)])


def scarf_points(pose, J, t, k):
    neck = J["neck"]
    u = J["u"]
    base = (neck[0] - J["sth"] * 0.05, neck[1] + u[1] * 0.02)
    wx, wy = pose.get("scarf_wind", (-1.0, 0.0))
    spd = pose.get("speed", 0.0)
    n = 9
    pts = [base]
    seg = 0.045 + 0.006 * k
    ang = math.atan2(wy + 0.9 - spd * 0.75, wx * (0.35 + spd))
    for i in range(1, n):
        wave = math.sin(t * (9 + spd * 8) - i * 0.9 + k * 1.3) * (0.25 + spd * 0.35) * (i / n)
        a = ang + wave
        p = pts[-1]
        pts.append((p[0] + math.cos(a) * seg, p[1] + math.sin(a) * seg))
    return pts


def _scarf_tails(ink, pose, J, t):
    for k in (0, 1):
        pts = scarf_points(pose, J, t, k)
        left, right = [], []
        for i, p in enumerate(pts):
            q = pts[min(i + 1, len(pts) - 1)] if i < len(pts) - 1 else pts[i]
            pr_ = pts[max(i - 1, 0)]
            dx, dy = q[0] - pr_[0], q[1] - pr_[1]
            L = math.hypot(dx, dy) or 1
            w = 0.028 * (1 - i / len(pts) * 0.35)
            left.append((p[0] - dy / L * w, p[1] + dx / L * w))
            right.append((p[0] + dy / L * w, p[1] - dx / L * w))
        tail = path_from(left + list(reversed(right)), close=True, smooth_k=0.3)
        ink.group([(tail, SCARF if k == 0 else scale_c(C(SCARF), 0.85), SCARF_SH)])


# ----------------------------------------------------------------------------
# animation routines
# ----------------------------------------------------------------------------
def run_pose(b, speed=1.0, yaw=0.97, stride=0.36, lift=0.16, steps_per_beat=2, t=0.0):
    """Sprint cycle; two steps per beat by default. Feet relative to pelvis (world scroll handles travel)."""
    p = base_pose(yaw)
    ph = (b * steps_per_beat / 2.0) % 1.0
    for sd, off in ((1, 0.0), (-1, 0.5)):
        f = (ph + off) % 1.0
        if f < 0.5:
            u_ = f / 0.5
            x = lerp(stride / 2, -stride / 2, u_)
            y = 0.0
            fa = 0.0 if u_ < 0.7 else -35 * (u_ - 0.7) / 0.3
        else:
            u_ = (f - 0.5) / 0.5
            e = ease_io(u_)
            x = lerp(-stride / 2, stride / 2, e) + math.sin(u_ * math.pi) * 0.02
            y = -lift * math.sin(u_ * math.pi) ** 0.8
            fa = lerp(-35, 10, u_)
        p["feet"][sd] = (x, y)
        p["foot_ang"][sd] = fa
    contact = abs(math.cos(ph * 2 * math.pi))
    p["root"] = (0.0, -PELVIS_H + 0.025 + 0.02 * contact)
    p["lean"] = 18.0 * speed
    # arms pump opposite to legs, elbows ~90deg
    for sd in (-1, 1):
        f = (ph + (0.5 if sd == 1 else 0.0)) % 1.0
        sw = math.cos(f * 2 * math.pi)
        p["hands"][sd] = (0.02 + sw * 0.12, -PELVIS_H - 0.05 - 0.05 * max(0.0, sw))
        p["elbow_hint"][sd] = (-1.0, 0.2)
    p["expr"] = "determined"
    p["hair_v"] = (-0.22 * speed, 0.05)
    p["speed"] = speed
    p["scarf_wind"] = (-1.0, 0.0)
    p["guitar"] = "back"
    p["hand_shape"] = {1: "fist", -1: "fist"}
    p["t"] = t
    return p


def play_pose(b, t, yaw=0.35, fret=0.5, strum_rate=2, intensity=1.0, stance=0.16, bang=1.0, note_age=1.0):
    """Standing and playing. Strum on the eighth notes, head-bang on the beat."""
    p = base_pose(yaw)
    p["guitar"] = "play"
    u_ = b % 1.0
    dip = 0.022 * intensity * math.exp(-u_ * 5)
    p["root"] = (0.0, -PELVIS_H + 0.02 + dip)
    p["feet"] = {1: (stance, 0.0), -1: (-stance * 0.8, 0.0)}
    p["foot_ang"] = {1: 8.0, -1: -5.0}
    p["lean"] = -4.0
    p["tilt"] = 2.5 * math.sin(b * math.pi) * intensity
    ph = (b * strum_rate) % 1.0
    p["strum"] = math.sin(ph * math.pi * 2)
    p["fret"] = fret
    p["head_nod"] = 0.03 * bang * intensity * math.exp(-u_ * 6)
    p["head_tilt"] = -6 * bang * intensity * math.exp(-u_ * 6)
    p["hair_bounce"] = 0.35 * bang * intensity * math.exp(-u_ * 5)
    p["expr"] = "rock"
    p["gtr_glow"] = clamp(1 - note_age * 3) * intensity
    p["string_vib"] = clamp(1 - note_age * 2)
    p["hand_shape"] = {1: "fret", -1: "pick"}
    p["t"] = t
    return p


def finish_play_hands(p):
    """Attach hands to the guitar for a 'play' pose (needs joints to place the guitar)."""
    J = joints(p)
    fx = lerp(0.47, 0.16, clamp(p["fret"]))
    p["hands"][1] = guitar_point(p, J, fx, 0.028)
    p["elbow_hint"][1] = (0.3, 1.0)
    sy = p["strum"] * 0.045
    p["hands"][-1] = guitar_point(p, J, -0.02, sy - 0.02)
    p["elbow_hint"][-1] = (-0.5, -1.0)
    return p


def pitch_to_fret(m):
    return clamp((m - 45) / 36.0)
