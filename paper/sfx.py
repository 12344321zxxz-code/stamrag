"""Synthesised sound effects and ambiences (numpy only), all mono float32 at SR unless noted."""
import numpy as np
from scipy.signal import butter, sosfilt

from .core import SR

RNG = np.random.default_rng(7)


def _t(d):
    return np.arange(int(d * SR)) / SR


def _env(n, a, d, curve=4.0):
    """Attack a seconds, then exponential decay over d seconds."""
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), np.exp(-curve * (t - a) / max(d, 1e-4)))
    return e.astype(np.float32)


def _bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x)


def _lp(x, f, order=2):
    return sosfilt(butter(order, f, "lowpass", fs=SR, output="sos"), x)


def _hp(x, f, order=2):
    return sosfilt(butter(order, f, "highpass", fs=SR, output="sos"), x)


def _sweep(f0, f1, d, shape=1.0):
    t = _t(d)
    u = (t / d) ** shape
    f = f0 * (f1 / f0) ** u
    return np.sin(2 * np.pi * np.cumsum(f) / SR), f


def _norm(x, peak=0.8):
    m = np.abs(x).max()
    return (x * (peak / m)).astype(np.float32) if m > 0 else x.astype(np.float32)


def crackle(d, density=90, amp=1.0):
    """Paper crackle: sparse little clicks through a bright band."""
    n = int(d * SR)
    x = np.zeros(n)
    k = RNG.poisson(density * d)
    idx = RNG.integers(0, n, k)
    x[idx] = RNG.uniform(-1, 1, k) * RNG.uniform(0.2, 1.0, k) ** 2
    x = _bp(x, 1800, 9000)
    return (x * amp).astype(np.float32)


def pop(pitch=1.0, d=0.16):
    s, _ = _sweep(900 * pitch, 380 * pitch, d, 0.35)
    click = _hp(RNG.standard_normal(int(0.006 * SR)), 3000) * 0.3
    x = s * _env(len(s), 0.002, d * 0.5, 5)
    x[:len(click)] += click
    return _norm(x, 0.7)


def drip(pitch=1.0):
    s, _ = _sweep(650 * pitch, 1500 * pitch, 0.07, 0.6)
    tail = np.zeros(int(0.25 * SR))
    tail[:len(s)] = s * _env(len(s), 0.001, 0.05, 3)
    return _norm(_lp(tail, 6000), 0.6)


def splash(size=1.0):
    d = 0.5 + 0.5 * size
    n = int(d * SR)
    noise = RNG.standard_normal(n)
    e = _env(n, 0.004, d * 0.45, 5)
    body = _bp(noise, 500, 6000) * e
    low = _lp(noise, 700) * _env(n, 0.002, 0.12, 5) * 1.6 * size
    x = body + low
    for _ in range(int(6 + 8 * size)):
        p = drip(RNG.uniform(0.8, 1.9)) * RNG.uniform(0.15, 0.4)
        o = RNG.integers(int(0.05 * SR), max(int(0.06 * SR), n - len(p)))
        x[o:o + len(p)] += p[: n - o]
    return _norm(x, 0.8)


def whoosh(d=0.7, f0=400, f1=2400, amp=1.0):
    n = int(d * SR)
    noise = RNG.standard_normal(n)
    t = np.arange(n) / n
    out = np.zeros(n)
    k = 8
    for i in range(k):
        a, b = i * n // k, (i + 1) * n // k
        fc = f0 * (f1 / f0) ** ((a + b) / 2 / n)
        seg = _bp(noise[max(0, a - 2048):b], fc * 0.6, min(fc * 1.7, 20000))[-(b - a):]
        out[a:b] = seg
    bell = np.sin(np.pi * t) ** 1.5
    x = out * bell + crackle(d, 60, 0.6) * bell
    return _norm(x, 0.55 * amp)


def whoosh_up(d=0.8):
    return whoosh(d, 350, 3200, 1.0)


def paper_slide(d=1.0):
    n = int(d * SR)
    x = _bp(RNG.standard_normal(n), 250, 4200) * np.sin(np.pi * np.arange(n) / n) ** 0.8
    x = x + crackle(d, 140, 0.9)
    return _norm(x, 0.5)


def sparkle(d=0.9, base=2093.0):
    n = int(d * SR)
    x = np.zeros(n)
    notes = [1, 5 / 4, 3 / 2, 2, 5 / 2, 3, 4]
    for i in range(14):
        f = base * notes[RNG.integers(0, len(notes))] * RNG.choice([0.5, 1, 1])
        o = int((i / 14) ** 1.3 * (d - 0.3) * SR)
        m = n - o
        t = np.arange(m) / SR
        tone = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t)) * np.exp(-t * 9)
        x[o:] += tone * RNG.uniform(0.3, 0.8)
    return _norm(x, 0.45)


def idea():
    """'Ding!': a bright bell with a little upward flourish."""
    x = np.zeros(int(1.0 * SR))
    for j, f in enumerate([1318.5, 1760.0, 2637.0]):
        o = int(j * 0.06 * SR)
        t = np.arange(len(x) - o) / SR
        x[o:] += (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 3.01 * t)) * np.exp(-t * 4.5) * (0.6 + 0.2 * j)
    return _norm(x, 0.5)


def boing(d=0.75):
    t = _t(d)
    f = 150 * (1 + 0.9 * t / d) * (1 + 0.35 * np.sin(2 * np.pi * 13 * t) * np.exp(-3.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)) * _env(len(t), 0.004, d * 0.6, 3.5)
    return _norm(_lp(x, 3000), 0.7)


def ribbit():
    out = []
    for syl, f0 in ((0.14, 190), (0.2, 150)):
        t = _t(syl)
        am = (np.sin(2 * np.pi * 34 * t) > -0.2).astype(float)
        saw = ((t * f0) % 1.0) * 2 - 1
        v = _bp(saw * am, 300, 1400) * np.sin(np.pi * t / syl)
        out += [v, np.zeros(int(0.05 * SR))]
    return _norm(np.concatenate(out), 0.6)


def peep(pitch=1.0):
    s, _ = _sweep(2300 * pitch, 3300 * pitch, 0.09, 0.8)
    return (s * np.sin(np.pi * np.arange(len(s)) / len(s)) ** 0.7 * 0.5).astype(np.float32)


def peeps(n=7, d=1.6):
    x = np.zeros(int(d * SR))
    for i in range(n):
        p = peep(RNG.uniform(0.85, 1.25))
        o = int((i + RNG.uniform(-0.2, 0.2)) / n * (d - 0.15) * SR)
        o = max(0, o)
        x[o:o + len(p)] += p
    return _norm(x, 0.35)


def tweet():
    parts = []
    for _ in range(RNG.integers(2, 5)):
        f0 = RNG.uniform(2800, 4200)
        s, _ = _sweep(f0, f0 * RNG.uniform(1.2, 1.6), RNG.uniform(0.04, 0.08), 1)
        parts += [s * np.hanning(len(s)), np.zeros(int(RNG.uniform(0.03, 0.07) * SR))]
    return (np.concatenate(parts) * 0.3).astype(np.float32)


def bubbles(n=8, d=1.2):
    x = np.zeros(int(d * SR))
    for i in range(n):
        f = RNG.uniform(350, 900)
        s, _ = _sweep(f, f * 2.1, RNG.uniform(0.03, 0.06), 0.5)
        s = s * _env(len(s), 0.002, 0.03, 3)
        o = int(RNG.uniform(0, d - 0.08) * SR)
        x[o:o + len(s)] += s * RNG.uniform(0.3, 0.7)
    return _norm(_lp(x, 4000), 0.5)


def whale_call(d=2.2):
    t = _t(d)
    f = 150 + 70 * np.sin(np.pi * t / d) ** 1.2 - 20 * t / d
    f = f * (1 + 0.012 * np.sin(2 * np.pi * 5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)) * np.sin(np.pi * t / d) ** 0.8
    return _norm(_lp(x, 1500), 0.55)


def whale_rise():
    a = whoosh(1.1, 120, 900, 0.9)
    b = splash(1.4)
    x = np.zeros(int(2.0 * SR))
    x[:len(a)] += a
    o = int(0.7 * SR)
    x[o:o + len(b)] += b[: len(x) - o] * 0.9
    return _norm(x, 0.85)


def spout(d=1.3):
    n = int(d * SR)
    x = _bp(RNG.standard_normal(n), 900, 9000) * _env(n, 0.05, d * 0.7, 3)
    x += _lp(RNG.standard_normal(n), 400) * _env(n, 0.01, 0.3, 4) * 1.5
    return _norm(x, 0.55)


def flutter(d=1.1):
    t = _t(d)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 28 * t)
    x = _bp(RNG.standard_normal(len(t)), 150, 1400) * am * np.sin(np.pi * t / d)
    return _norm(x, 0.35)


def paddle():
    """Boat setting off: a soft splash and a lapping wake."""
    x = np.zeros(int(1.4 * SR))
    for i, o in enumerate((0.0, 0.35, 0.7)):
        s = splash(0.25) * (0.5 - 0.1 * i)
        k = int(o * SR)
        x[k:k + len(s)] += s[: len(x) - k]
    return _norm(x, 0.45)


def tongue():
    s, _ = _sweep(300, 1800, 0.12, 0.7)
    x = s * _env(len(s), 0.003, 0.08, 3) + _hp(RNG.standard_normal(len(s)), 2500) * 0.15 * _env(len(s), 0.001, 0.03, 4)
    return _norm(x, 0.5)


def confetti(d=2.5):
    x = crackle(d, 420, 1.0) * np.exp(-np.arange(int(d * SR)) / SR * 1.2)
    pops = np.zeros_like(x)
    for i in range(6):
        p = pop(RNG.uniform(0.9, 1.6), 0.1) * 0.5
        o = int(RNG.uniform(0, 0.6) * SR)
        pops[o:o + len(p)] += p
    return _norm(x + pops, 0.5)


# -- ambiences (stereo, loopable-ish) -------------------------------------------------------------------
def _pink(n):
    w = RNG.standard_normal(n)
    f = np.fft.rfft(w)
    k = np.arange(len(f))
    k[0] = 1
    return np.fft.irfft(f / np.sqrt(k), n)


def amb_water(d, kind="stream"):
    n = int(d * SR)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        if kind == "sea":
            base = _lp(_pink(n), 900)
            t = np.arange(n) / SR
            lfo = 0.55 + 0.45 * np.sin(2 * np.pi * (0.11 + 0.02 * ch) * t + ch) ** 2
            x = base * lfo + _bp(RNG.standard_normal(n), 2500, 7000) * 0.04 * lfo
        else:
            x = _bp(_pink(n), 400, 3000) * 0.6
            for _ in range(int(d * (9 if kind == "stream" else 3))):
                f = RNG.uniform(500, 1400)
                s, _ = _sweep(f, f * 1.6, 0.05, 0.6)
                s = s * _env(len(s), 0.003, 0.04, 3) * RNG.uniform(0.02, 0.06)
                o = RNG.integers(0, n - len(s))
                x[o:o + len(s)] += s * np.abs(x).max()
        out[:, ch] = x
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def amb_birds(d, density=0.6):
    n = int(d * SR)
    out = np.zeros((n, 2), np.float32)
    t = 0.3
    while t < d - 0.5:
        s = tweet()
        pan = RNG.uniform(0.15, 0.85)
        o = int(t * SR)
        m = min(len(s), n - o)
        out[o:o + m, 0] += s[:m] * (1 - pan)
        out[o:o + m, 1] += s[:m] * pan
        t += RNG.exponential(1 / density) + 0.2
    return out


SFX = dict(pop=pop, drip=drip, splash=splash, whoosh=whoosh, whoosh_up=whoosh_up, paper_slide=paper_slide,
           sparkle=sparkle, idea=idea, boing=boing, ribbit=ribbit, peeps=peeps, bubbles=bubbles,
           whale_call=whale_call, whale_rise=whale_rise, spout=spout, flutter=flutter, paddle=paddle,
           tongue=tongue, confetti=confetti)
