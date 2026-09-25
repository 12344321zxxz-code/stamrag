"""Builds the whole soundtrack and the timeline the animation is cut to.

    python -m paper.audio [--check]

1. every line is spoken by the cast (voice.py, cached)          -> build_paper/voice/
2. the cue sheet is laid out on a bar grid of exactly 2:00      -> build_paper/timeline.json
3. the score is written and rendered (music.py)                 -> build_paper/music/
4. voices, music (ducked under speech), effects and ambiences are mixed, loudness-normalised to
   -16 LUFS                                                      -> build_paper/soundtrack.wav
5. per-line mouth envelopes for lip-sync                        -> build_paper/mouths.npz
--check round-trips every line through Whisper and prints what it hears.
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt, sosfiltfilt

from . import music, script, sfx, voice
from .core import BUILD, SR, build_path
from .timeline import layout, title_letter_times

PAN = dict(narrator=0.0, pip=-0.12, lulu=0.25, finn=0.25, duck1=0.05, duck2=0.25, duck3=0.4, freddy=0.25,
           whale=0.15, olive=0.3)
GROUP_OFFSET = 0.045
AMBIENCE = dict(title=None, garden="birds", red="stream", orange="stream", yellow="pond", green="pond",
                blue="sea", purple="sea", finale="sea")


def _texts():
    out = []
    for _, cues in script.SCENES:
        for c in cues:
            if c[0] == "say":
                out += [(m, c[2]) for m in script.GROUPS.get(c[1], [c[1]])]
            elif c[0] == "beats":
                out += [(m, w) for m in script.GROUPS.get(c[1], [c[1]]) for w in c[2]]
    return out


def _reverb_ir(rt60=0.45, wet=1.0, seed=3):
    rng = np.random.default_rng(seed)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir = sosfilt(butter(1, 5000, "lowpass", fs=SR, output="sos"), ir)
    ir[: int(0.012 * SR)] = 0
    return (ir / np.sqrt((ir ** 2).sum()) * wet).astype(np.float32)


def _place(bus, x, t, gain=1.0, pan=0.0):
    """Add mono or stereo x into the stereo bus at time t with an equal-power pan."""
    i = int(round(t * SR))
    if i >= len(bus):
        return
    if i < 0:
        x = x[-i:]
        i = 0
    n = min(len(x), len(bus) - i)
    if x.ndim == 1:
        a = (pan + 1) * np.pi / 4
        bus[i:i + n, 0] += x[:n] * gain * np.cos(a)
        bus[i:i + n, 1] += x[:n] * gain * np.sin(a)
    else:
        bus[i:i + n] += x[:n] * gain


def _envelope(x, fps=100):
    hop = SR // fps
    n = len(x) // hop + 1
    xp = np.pad(x, (0, n * hop - len(x) + hop))
    fr = xp[: n * hop].reshape(n, hop)
    e = np.sqrt((fr ** 2).mean(1))
    e = e / (np.percentile(e, 92) + 1e-9)
    e = np.clip((e - 0.12) / 0.88, 0, 1.2)
    k = np.array([0.25, 0.5, 0.25])
    return np.convolve(e, k, mode="same").astype(np.float32)


def _follow(x, att, rel, fps):
    out = np.zeros_like(x)
    v = 0.0
    a = 1 - np.exp(-1 / (att * fps))
    r = 1 - np.exp(-1 / (rel * fps))
    for i, s in enumerate(x):
        v += (s - v) * (a if s > v else r)
        out[i] = v
    return out


def loudness(x):
    """Integrated loudness (LUFS, BS.1770 K-weighting with gating) of a stereo float array."""
    from scipy.signal import lfilter
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    y = lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    ms = np.array([(y[i:i + blk] ** 2).mean(0).sum() for i in range(0, len(y) - blk, hop)])
    lk = -0.691 + 10 * np.log10(ms + 1e-12)
    g = ms[lk > -70]
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g = ms[(lk > -70) & (lk > rel)]
    return -0.691 + 10 * np.log10(g.mean())


def limiter(x, ceiling=0.89, release=0.08):
    """Look-ahead peak limiter: instant (look-ahead) attack, exponential release, 1 ms control rate."""
    from scipy.ndimage import maximum_filter1d
    blk = SR // 1000
    need = np.maximum(1.0, np.abs(x).max(1) / ceiling)
    nb = len(need) // blk + 1
    need = np.pad(need, (0, nb * blk - len(need)), constant_values=1.0).reshape(nb, blk).max(1)
    need = maximum_filter1d(need, size=9)
    g = _follow(1.0 / need, 1e-4, release, 1000)
    g = np.minimum(g, 1.0 / need)
    gs = np.interp(np.arange(len(x)) / blk, np.arange(nb), g)
    return x * gs[:, None]


def build(check=False):
    os.makedirs(BUILD, exist_ok=True)
    print("voices ...", flush=True)
    clips = {}
    for m, text in _texts():
        if (m, text) not in clips:
            clips[(m, text)] = voice.render(m, text)
    if check:
        voice.check([(m, t, x) for (m, t), x in clips.items()])

    tl = layout(lambda m, text: len(clips[(m, text)]) / SR)
    print(f"tempo {tl['tempo']:.2f} bpm, {tl['n_bars']} bars, bar {tl['bar']:.3f}s", flush=True)
    for sc in tl["scenes"]:
        print(f"  {sc['name']:8s} {sc['t0']:7.2f} - {sc['t1']:7.2f}  ({sc['bars']} bars)")

    n = int(tl["duration"] * SR)
    vox = np.zeros((n, 2), np.float32)
    ir_room, ir_big = _reverb_ir(0.35), _reverb_ir(1.6, seed=5)
    mouths = {}
    for li, ln in enumerate(tl["lines"]):
        members = ln["members"]
        ln["offsets"] = {}
        for k, m in enumerate(members):
            x = clips[(m, ln["text"])]
            off = 0.0 if len(members) == 1 else GROUP_OFFSET * k * 0.5
            g = 1.0 if len(members) == 1 else 0.7
            if ln["text"].rstrip("!").lower() in script.COLORS:
                g *= 1.25
            wet = fftconvolve(x, ir_big if m == "whale" else ir_room)[: len(x) + int(0.4 * SR)]
            dry = np.pad(x, (0, len(wet) - len(x)))
            y = dry + wet * (0.16 if m == "whale" else 0.07)
            _place(vox, y, ln["t0"] + off, g, PAN.get(m, 0.0))
            ln["offsets"][m] = off
            mouths[f"{li}:{m}"] = _envelope(x)
    np.savez_compressed(build_path("mouths.npz"), **mouths)

    print("effects ...", flush=True)
    fx = np.zeros((n, 2), np.float32)
    for e in tl["sfx"]:
        _place(fx, sfx.SFX[e["name"]](**e["kw"]), e["t"], 0.8, 0.1)
    for i, t in enumerate(title_letter_times(tl)):
        _place(fx, sfx.pop(1.0 + 0.06 * (i % 7)), t, 0.55, -0.4 + 0.04 * i)
    for i in range(6):
        _place(fx, sfx.pop(1.1 + 0.1 * i, 0.12), tl["marks"]["garden.tracker"] + 0.15 + 0.13 * i, 0.5, -0.6)
    for c in script.COLORS:
        _place(fx, sfx.pop(1.5, 0.14), tl["marks"][f"{c}.word"] + 1.0, 0.55, -0.6)
    for i in range(6):
        _place(fx, sfx.whoosh(0.5, 500 + 200 * i, 2500 + 300 * i, 0.8), tl["marks"][f"finale.beat{i}"] - 0.1, 0.45,
               -0.5 + 0.2 * i)
    for sc in tl["scenes"][1:]:
        f = sfx.paper_slide(1.1) if sc["name"] == "garden" else sfx.whoosh(0.8, 300, 2600, 1.0)
        _place(fx, f, sc["t0"] - 0.45, 0.7, 0.0)
    # ambiences, cross-faded per scene
    amb = np.zeros((n, 2), np.float32)
    for sc in tl["scenes"]:
        kind = AMBIENCE[sc["name"]]
        if kind is None:
            continue
        d = sc["t1"] - sc["t0"] + 1.0
        if kind == "birds":
            a = sfx.amb_birds(d, 0.5) * 0.9 + sfx.amb_water(d, "pond") * 0.12
        elif kind == "stream":
            a = sfx.amb_water(d, "stream") * 0.5 + sfx.amb_birds(d, 0.35) * 0.6
        elif kind == "pond":
            a = sfx.amb_water(d, "pond") * 0.35 + sfx.amb_birds(d, 0.3) * 0.6
        else:
            a = sfx.amb_water(d, "sea") * 0.7
        m = len(a)
        fade = np.minimum(1, np.minimum(np.arange(m), m - np.arange(m)) / (0.5 * SR))[:, None]
        _place(amb, a * fade, sc["t0"] - 0.5, 0.33)

    print("music ...", flush=True)
    mus, msr = sf.read(music.render(tl), dtype="float32", always_2d=True)
    assert msr == SR
    mus = np.pad(mus, ((0, max(0, n - len(mus))), (0, 0)))[:n]
    fps = 200
    ve = np.abs(vox).max(1)
    ve = ve[: len(ve) // (SR // fps) * (SR // fps)].reshape(-1, SR // fps).max(1)
    duck = _follow((ve > 0.02).astype(float), 0.03, 0.4, fps)
    d = np.interp(np.arange(n) / SR, np.arange(len(duck)) / fps, duck)[:, None]
    # under speech: -11 dB overall and a further -6 dB across the speech band (zero-phase band cut)
    mid = sosfiltfilt(butter(2, [500, 5000], "bandpass", fs=SR, output="sos"), mus, axis=0)
    mus = (mus - 0.5 * d * mid) * (1.0 - 0.72 * d)
    mus = mus / (np.abs(mus).max() + 1e-9) * 10 ** (-3 / 20)
    tail = int(0.9 * SR)
    mus[-tail:] *= np.linspace(1, 0, tail)[:, None] ** 1.5

    mix = vox * 1.0 + mus * 0.5 + fx * 0.5 + amb * 0.4
    mix = sosfilt(butter(2, 30, "highpass", fs=SR, output="sos"), mix, axis=0)
    lufs = loudness(mix)
    mix = mix * 10 ** ((-16.0 - lufs) / 20)
    mix = limiter(mix, 0.89)
    print(f"loudness {lufs:.1f} -> {loudness(mix):.1f} LUFS, peak {np.abs(mix).max():.3f}", flush=True)
    sf.write(build_path("soundtrack.wav"), mix.astype(np.float32), SR, subtype="PCM_16")
    for name, bus in (("stem_voice", vox), ("stem_music", mus), ("stem_fx", fx + amb)):
        sf.write(build_path("stems", name + ".wav"), bus.astype(np.float32), SR, subtype="PCM_16")
    with open(build_path("timeline.json"), "w") as fh:
        json.dump(tl, fh, indent=1)
    print("wrote", build_path("soundtrack.wav"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    build(a.check)


if __name__ == "__main__":
    sys.exit(main())
