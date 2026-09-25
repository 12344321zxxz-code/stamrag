"""The score: an original, bar-aligned kids' underscore written as MIDI and rendered with FluidSynth.

Written in C and transposed to F. Every scene has its own band (glockenspiel for the ladybug, marimba
for the fish, oboe and bassoon for the ducklings, tuba and woodblocks for the frog, horn, harp and strings
for the whale, xylophone for the octopus). The melody thins out under speech, each colour word lands
on a V-I cadence with a glockenspiel 'ta-da', and the finale climbs one scale step per rainbow arc.
"""
import math
import os
import subprocess

import mido

from .core import SR, build_path

KEY = 5  # C -> F
SOUNDFONTS = ["/usr/share/sounds/sf3/MuseScore_General.sf3", "/usr/share/sounds/sf2/FluidR3_GM.sf2"]

# chord = (root pitch class, quality)
C, Dm, Em, F, G, G7, Am = (0, "M"), (2, "m"), (4, "m"), (5, "M"), (7, "M"), (7, "7"), (9, "m")
QUAL = {"M": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10)}

# the theme (chords C F G C Am F G C): (beat, beats, midi)
THEME = [
    [(0, 1, 76), (1, .5, 72), (1.5, .5, 74), (2, 1, 76), (3, 1, 79)],
    [(0, 1, 81), (1, .5, 79), (1.5, .5, 77), (2, 2, 77)],
    [(0, 1, 74), (1, .5, 76), (1.5, .5, 77), (2, 1, 79), (3, 1, 74)],
    [(0, 1, 76), (1, 1, 74), (2, 2, 72)],
    [(0, 1, 72), (1, .5, 76), (1.5, .5, 81), (2, 1, 79), (3, 1, 76)],
    [(0, 1, 77), (1, .5, 81), (1.5, .5, 79), (2, 1, 77), (3, 1, 72)],
    [(0, 1, 74), (1, .5, 79), (1.5, .5, 77), (2, 1, 76), (3, 1, 74)],
    [(0, 3, 72)],
]
THEME_CHORDS = [C, F, G, C, Am, F, G, C]
# motif for any chord: a theme bar of the same quality, moved to the chord's root
MOTIF_SRC = {"M": [(0, 0), (5, 1), (7, 2)], "7": [(7, 6)], "m": [(9, 4)]}

LOOPS = {
    "red": [C, Am, F, G], "orange": [C, F, C, G], "yellow": [F, C, G, C], "green": [C, G, C, G],
    "blue": [F, C, Dm, G], "purple": [Am, F, C, G],
}

# GM programs per role; None = tacet
STYLE = {
    "title": dict(mel=(9, 73), chord=24, bass=58, arp=None, pad=None, drums="full"),
    "grey": dict(mel=None, chord=None, bass=45, arp=8, pad=49, drums=None),
    "bright": dict(mel=(9, None), chord=None, bass=32, arp=45, pad=None, drums="light"),
    "sail": dict(mel=(9, 73), chord=24, bass=58, arp=45, pad=None, drums="full"),
    "red": dict(mel=(9, None), chord=None, bass=32, arp=45, pad=None, drums="light"),
    "orange": dict(mel=(12, None), chord=None, bass=32, arp=12, pad=None, drums="light"),
    "yellow": dict(mel=(68, None), chord=24, bass=70, arp=None, pad=None, drums="light"),
    "green": dict(mel=(71, None), chord=24, bass=58, arp=None, pad=None, drums="frog"),
    "blue": dict(mel=(60, None), chord=None, bass=42, arp=46, pad=49, drums=None),
    "purple": dict(mel=(13, 71), chord=None, bass=32, arp=45, pad=None, drums="light"),
    "finale": dict(mel=(9, 73), chord=24, bass=58, arp=45, pad=48, drums="full"),
}
CH = dict(mel1=0, mel2=1, chord=2, bass=3, arp=4, pad=5, harp=6, spark=7, brass=8, drums=9, choir=10, sting=11,
          timp=12)
MIX = dict(mel1=(100, 64, 50), mel2=(78, 44, 50), chord=(70, 84, 40), bass=(96, 64, 30), arp=(74, 40, 45),
           pad=(62, 64, 60), harp=(80, 90, 60), spark=(72, 30, 60), brass=(76, 70, 55), choir=(78, 64, 70),
           sting=(92, 64, 60), timp=(90, 64, 40))
FIXED_PROG = dict(harp=46, spark=8, brass=60, choir=52, sting=9, timp=47)


def tones(ch, octave_base):
    root, q = ch
    return [octave_base + root + i for i in QUAL[q]]


SCALE = [0, 2, 4, 5, 7, 9, 11]


def dshift(p, steps):
    """Move a (C major) pitch by diatonic scale steps, so motifs stay in key."""
    o, pc = divmod(p, 12)
    d = SCALE.index(pc) + steps
    return (o + d // 7) * 12 + SCALE[d % 7]


def motif(ch, k=0):
    root, q = ch
    src_root, idx = MOTIF_SRC[q][k % len(MOTIF_SRC[q])]
    steps = SCALE.index(root) - SCALE.index(src_root)
    steps = steps - 7 if steps > 3 else steps + 7 if steps < -3 else steps
    return [(b, l, dshift(p, steps)) for (b, l, p) in THEME[idx]]


class Score:
    def __init__(self, tl):
        self.tl = tl
        self.beat = tl["beat"]
        self.ev = []
        self.speech = [(ln["t0"], ln["t1"]) for ln in tl["lines"]]

    def t(self, beat):
        return beat * self.beat

    def talking(self, b0, b1, pad=0.12):
        t0, t1 = self.t(b0) - pad, self.t(b1) + pad
        return any(a < t1 and b > t0 for a, b in self.speech)

    def note(self, role, beat, dur, pitch, vel, transpose=True):
        ch = CH[role]
        p = pitch + (KEY if transpose and ch != 9 else 0)
        vel = int(max(1, min(127, vel)))
        self.ev.append((beat, 1, mido.Message("note_on", channel=ch, note=int(p), velocity=vel)))
        self.ev.append((beat + max(0.05, dur), 0, mido.Message("note_off", channel=ch, note=int(p), velocity=0)))

    def prog(self, role, beat, program):
        self.ev.append((beat, -2, mido.Message("program_change", channel=CH[role], program=program)))

    def cc(self, role, beat, num, val):
        self.ev.append((beat, -1, mido.Message("control_change", channel=CH[role], control=num, value=int(val))))

    def write(self, path):
        mid = mido.MidiFile(ticks_per_beat=480, type=0)
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=int(round(self.beat * 1e6)), time=0))
        last = 0
        for beat, _, msg in sorted(self.ev, key=lambda e: (e[0], e[1])):
            tick = int(round(beat * 480))
            tr.append(msg.copy(time=tick - last))
            last = tick
        mid.save(path)


# ---------------------------------------------------------------------------------------------------
def _drums(s, style, b0, busy):
    k = 0.55 if busy else 1.0
    for i in range(8):
        b = b0 + i * 0.5
        if style in ("full", "light", "frog"):
            s.note("drums", b, 0.2, 70, (44 if i % 2 == 0 else 30) * (0.8 if style != "full" else 1.0) * k)
        if style == "full":
            if i in (0, 4):
                s.note("drums", b, 0.2, 36, 78 * k)
            if i in (2, 6):
                s.note("drums", b, 0.2, 39, 58 * k)
            if i % 2 == 1:
                s.note("drums", b, 0.2, 54, 26 * k)
        elif style == "light" and i in (2, 6):
            s.note("drums", b, 0.2, 76, 40 * k)
        elif style == "frog":
            if i in (0, 4):
                s.note("drums", b, 0.2, 77, 62 * k)
            if i in (3, 7):
                s.note("drums", b, 0.2, 76, 52 * k)


def _accomp(s, st, chords, b0, busy, scene):
    """chords: [(beat_in_bar, chord)] covering the bar."""
    segs = [(cb, chords[i + 1][0] if i + 1 < len(chords) else 4, ch) for i, (cb, ch) in enumerate(chords)]
    half = scene == "blue"
    for cb, ce, ch in segs:
        root = ch[0]
        bass_root = 36 + root if root < 5 else 24 + root
        fifth = bass_root + 7
        if st["bass"] is not None:
            if half:
                s.note("bass", b0 + cb, ce - cb, bass_root, 72)
            else:
                for j, bb in enumerate(range(int(cb), int(math.ceil(ce)))):
                    if bb % 2 == 0:
                        s.note("bass", b0 + bb, 0.9 if st["bass"] != 45 else 0.5,
                               bass_root if (bb // 2) % 2 == 0 else fifth, 84 if bb == 0 else 70)
        if st["chord"] is not None:
            for bb in range(int(cb), int(math.ceil(ce))):
                if bb % 2 == 1:
                    for p in tones(ch, 60):
                        s.note("chord", b0 + bb, 0.35, p if p < 72 else p - 12, 58)
        if st["arp"] is not None:
            tt = tones(ch, 60) + [60 + root + 12]
            pat = [0, 1, 2, 3, 2, 1, 2, 1] if not half else [0, 2, 3, 1, 2, 3, 1, 2]
            step = 0.5 if st["arp"] not in (8,) else 1.0
            x = cb
            while x < ce - 1e-6:
                idx = pat[int(round(x / step)) % len(pat)]
                p = tt[idx % len(tt)]
                s.note("arp", b0 + x, step * (1.6 if st["arp"] in (46, 8) else 0.8), p + (12 if st["arp"] == 8 else 0),
                       (46 if busy else 58) * (0.8 if st["arp"] == 8 else 1.0))
                x += step
        if st["pad"] is not None:
            for p in tones(ch, 48 + (12 if root < 5 else 0)):
                s.note("pad", b0 + cb, ce - cb, p, 50 if busy else 62)


def _melody(s, st, chords, b0, k, scene, theme_bar=None):
    if st["mel"] is None:
        return
    if theme_bar is not None:
        notes = THEME[theme_bar]
    elif len(chords) == 1:
        notes = motif(chords[0][1], k)
    else:
        notes = [(cb, (chords[i + 1][0] if i + 1 < len(chords) else 4) - cb, tones(ch, 72)[1])
                 for i, (cb, ch) in enumerate(chords)]
    if scene == "blue":
        notes = [(b, l, p - 12) for (b, l, p) in notes if b in (0, 2)]
        notes = [(b, 2, p) for (b, l, p) in notes]
    for b, l, p in notes:
        quiet = s.talking(b0 + b, b0 + b + l)
        v = 84 if not quiet else 30
        s.note("mel1", b0 + b, l * 0.92, p, v)
        if st["mel"][1] is not None:
            s.note("mel2", b0 + b, l * 0.92, p - 12, v * 0.8)


def _sting(s, beat):
    for i, p in enumerate((72, 76, 79, 84)):
        s.note("sting", beat + i * 0.25, 0.9, p + 12, 92 - 6 * i)
    s.note("drums", beat, 1.0, 81, 70)
    s.note("drums", beat, 1.0, 49, 40)


def _gliss(s, beat, beats=1.0, up=True, vel=62):
    scale = [0, 2, 4, 5, 7, 9, 11]
    ps = [60 + 12 * o + d for o in range(2) for d in scale] + [84]
    if not up:
        ps = ps[::-1]
    n = len(ps)
    for i, p in enumerate(ps):
        s.note("harp", beat + beats * i / n, 0.6, p, vel * (0.7 + 0.3 * i / n))


def compose(tl):
    s = Score(tl)
    for role, (vol, pan, rev) in MIX.items():
        s.cc(role, 0, 7, vol)
        s.cc(role, 0, 10, pan)
        s.cc(role, 0, 91, rev)
        s.cc(role, 0, 93, 10)
    for role, prog in FIXED_PROG.items():
        s.prog(role, 0, prog)
    s.cc("drums", 0, 7, 88)
    s.cc("drums", 0, 91, 30)
    marks = tl["marks"]
    beat = tl["beat"]

    def mbeat(key):
        return marks[key] / beat

    cur = {}

    def set_style(st, b):
        for role in ("mel1", "mel2", "chord", "bass", "arp", "pad"):
            prog = (st["mel"][0 if role == "mel1" else 1] if st["mel"] else None) if role.startswith("mel") else st[role]
            if prog is not None and cur.get(role) != prog:
                s.prog(role, b - 0.01 if b > 0 else 0, prog)
                cur[role] = prog

    for sc in tl["scenes"]:
        name, n, bar0 = sc["name"], sc["bars"], sc["bar0"]
        chords_by_bar = []
        if name == "title":
            chords_by_bar = [[(0, THEME_CHORDS[i])] for i in range(n)]
        elif name == "garden":
            idea_bar = next((i for i in range(n) if (bar0 + i) * 4 * beat >= marks["garden.idea"] - 0.2), n)
            chords_by_bar = [[(0, [Am, F, C, G][i % 4] if i < idea_bar else [C, F, G, C][(i - idea_bar) % 4])]
                             for i in range(n)]
        elif name in LOOPS:
            loop = LOOPS[name]
            chords_by_bar = [[(0, loop[i % 4])] for i in range(n)]
            wb = mbeat(f"{name}.word") - bar0 * 4
            wbar, wbeat = int(wb // 4), wb % 4
            if wbeat >= 2:
                chords_by_bar[wbar] = [(0, chords_by_bar[wbar][0][1]), (wbeat - 2, G), (wbeat, C)] if wbeat > 2 else \
                    [(0, G), (wbeat, C)]
            else:
                if wbar > 0:
                    chords_by_bar[wbar - 1] = [(0, chords_by_bar[wbar - 1][0][1]), (2 + wbeat, G)]
                chords_by_bar[wbar] = [(0, G), (wbeat, C)] if wbeat > 0 else [(0, C)]
            for i in range(wbar + 1, n):
                chords_by_bar[i] = [(0, [C, F, G, C][(i - wbar - 1) % 4])]
            _sting(s, mbeat(f"{name}.word") + 1)
            _gliss(s, mbeat(f"{name}.give"), 1.2, True, 58)
        elif name == "finale":
            arcs_bar = int(round(mbeat("finale.arcs"))) // 4 - bar0
            hooray_bar = int(round(mbeat("finale.hooray"))) // 4 - bar0
            end_bar = int(round(mbeat("finale.end"))) // 4 - bar0
            for i in range(n):
                if i < arcs_bar:
                    pre = ([C] * max(0, arcs_bar - 4) + [C, Am, F, G])[-arcs_bar:]
                    chords_by_bar.append([(0, pre[i])])
                elif i < hooray_bar:
                    j = (i - arcs_bar) * 2
                    seq = [C, Dm, Em, F, G, G7]
                    chords_by_bar.append([(0, seq[min(j, 5)]), (2, seq[min(j + 1, 5)])])
                elif i < end_bar:
                    k = i - hooray_bar
                    chords_by_bar.append([(0, THEME_CHORDS[k % 8] if i < end_bar - 1 else G)])
                else:
                    chords_by_bar.append([(0, C)])
        # -- play the scene ---------------------------------------------------------------------------
        for i in range(n):
            b0 = (bar0 + i) * 4
            chords = chords_by_bar[i]
            busy = s.talking(b0, b0 + 4)
            if name == "garden":
                phase = "grey" if i < idea_bar else (
                    "bright" if b0 * beat < marks["garden.sail"] - beat * 2 else "sail")
                st = STYLE[phase]
            else:
                st = STYLE[name]
            set_style(st, b0)
            if name == "finale" and i >= end_bar:
                _finale_end(s, b0, tl)
                continue
            _accomp(s, st, chords, b0, busy, name)
            if st["drums"]:
                _drums(s, st["drums"], b0, busy)
            theme_bar = i % 8 if name == "title" else None
            if name == "finale" and arcs_bar <= i < hooray_bar:
                pass
            elif name == "finale" and hooray_bar <= i < end_bar:
                k = i - hooray_bar
                if i < end_bar - 1:
                    _melody(s, st, chords, b0, 0, name, theme_bar=k % 8)
                else:
                    _melody(s, st, chords, b0, 0, name)
                for p in tones(chords[0][1], 60):
                    s.note("brass", b0, 3.8, p, 50)
            else:
                _melody(s, st, chords, b0, i, name, theme_bar)
        if name == "finale":
            # the climb: one scale step per rainbow arc, then 'Hooray!' on the tonic
            for j, p in enumerate((72, 74, 76, 77, 79, 83)):
                b = mbeat(f"finale.beat{j}")
                s.note("sting", b + 0.5, 1.4, p + 12, 74)
                s.note("mel1", b, 1.8, p, 66)
                s.note("drums", b, 0.3, 36, 70)
                for q in tones([C, Dm, Em, F, G, G7][j], 60):
                    s.note("harp", b + 0.5, 1.2, q, 52)
                    s.note("pad", b, 1.9, q - 12, 52)
            hb = mbeat("finale.hooray")
            s.note("drums", hb, 2.0, 49, 72)
            s.note("drums", hb, 0.3, 36, 90)
            for q in (60, 64, 67, 72):
                s.note("choir", hb, 7.5, q, 70)
            _gliss(s, hb, 1.0, True, 80)
            # snare roll under 'Up, up, up you go!'
            ub, lb = mbeat("finale.up"), mbeat("finale.arcs")
            x = ub
            while x < lb - 0.1:
                s.note("drums", x, 0.1, 38, 14 + 36 * (x - ub) / max(lb - ub, 1e-3))
                x += 0.25
            _gliss(s, lb - 1.0, 1.0, True, 70)
        # transition flourish into the next scene
        if name != "finale":
            _gliss(s, (bar0 + n) * 4 - 1.0, 0.9, True, 54)
    # whale reveal: timpani swell
    wb = mbeat("blue.reveal")
    x = wb
    while x < wb + 2.0:
        s.note("timp", x, 0.2, 36, 30 + 40 * (x - wb) / 2.0)
        x += 0.125
    return s


def _finale_end(s, b0, tl):
    total_beats = tl["duration"] / tl["beat"]
    d = total_beats - b0
    for p in (48, 55, 60, 64, 67, 72):
        s.note("pad", b0, d, p, 80)
        s.note("brass", b0, d, p, 70) if p >= 55 else None
    s.note("bass", b0, d, 36, 96)
    for p in (72, 76, 79, 84):
        s.note("mel1", b0, d, p, 80)
        s.note("choir", b0, d, p - 12, 76)
    s.note("drums", b0 + 1.5, 3, 49, 90)
    s.note("drums", b0, 0.3, 36, 90)
    s.note("drums", b0 + 1.5, 3, 81, 70)
    _gliss(s, b0 + 0.5, 1.0, True, 76)
    for i, p in enumerate((84, 88, 91, 96, 91, 96)):
        s.note("sting", b0 + 1.5 + i * 0.25, 1.5, p, 70 - 6 * i)


def render(tl):
    """-> path of the rendered stereo wav (SR)."""
    mid = build_path("music", "score.mid")
    wav = build_path("music", "score.wav")
    compose(tl).write(mid)
    sf = next(p for p in SOUNDFONTS if os.path.exists(p))
    subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.45", "-r", str(SR), "-o", "synth.reverb.room-size=0.55",
                    "-o", "synth.reverb.level=0.6", "-F", wav, sf, mid], check=True)
    return wav
