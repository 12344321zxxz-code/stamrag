"""Lays the cue sheet (script.py) out in time on a constant-tempo bar grid, and reads it back.

The layout needs every spoken line's duration, so it runs inside the audio build; the result is
written to build_paper/timeline.json, which the renderer loads through `timeline()`.
"""
import json
import math
import os

from . import script
from .core import BUILD

TEMPO_RANGE = (96.0, 124.0)
SQUEEZE = 0.6
SLACK_ORDER = ["finale", "garden", "title", "blue", "red", "orange", "yellow", "green", "purple"]


def _members(who):
    return script.GROUPS.get(who, [who])


def _layout_scene(name, cues, t_scene, beat, dur, slack=0.0, squeeze=1.0):
    """Returns (length, lines, marks, sfx) with times relative to the start of the film."""
    t = 0.0
    lines, marks, sfx = [], {}, []
    for cue in cues:
        kind = cue[0]
        if kind == "gap":
            t += cue[1] * squeeze
        elif kind == "hold":
            t += cue[1]
        elif kind == "wait_beats":
            t += cue[1] * beat
        elif kind == "slack":
            t += slack
        elif kind == "mark":
            marks[f"{name}.{cue[1]}"] = t_scene + t
        elif kind == "align":
            step = beat * (4 if cue[1] == "bar" else 1)
            t = math.ceil((t_scene + t) / step - 1e-6) * step - t_scene
        elif kind == "sfx":
            sfx.append(dict(name=cue[1], t=t_scene + t + cue[2], kw=cue[3]))
        elif kind == "say":
            who, text = cue[1], cue[2]
            d = max(dur(m, text) for m in _members(who))
            lines.append(dict(scene=name, who=who, members=_members(who), text=text, t0=t_scene + t,
                              t1=t_scene + t + d))
            t += d
        elif kind == "beats":
            who, words, n = cue[1], cue[2], cue[3]
            t = math.ceil((t_scene + t) / beat - 1e-6) * beat - t_scene
            for i, wd in enumerate(words):
                d = max(dur(m, wd) for m in _members(who))
                marks[f"{name}.beat{i}"] = t_scene + t
                lines.append(dict(scene=name, who=who, members=_members(who), text=wd, t0=t_scene + t,
                                  t1=t_scene + t + d))
                t += max(d, n * beat) if i < len(words) - 1 else d
        else:
            raise ValueError(cue)
    return t, lines, marks, sfx


def _fit(name, cues, t_scene, beat, dur, k, slack=0.0):
    """Largest gap squeeze in [SQUEEZE, 1] that fits the scene into k bars (None if impossible)."""
    cap = k * 4 * beat + 1e-6
    if _layout_scene(name, cues, t_scene, beat, dur, slack, SQUEEZE)[0] > cap:
        return None
    if _layout_scene(name, cues, t_scene, beat, dur, slack, 1.0)[0] <= cap:
        return 1.0
    lo, hi = SQUEEZE, 1.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if _layout_scene(name, cues, t_scene, beat, dur, slack, mid)[0] <= cap:
            lo = mid
        else:
            hi = mid
    return lo


def layout(dur):
    """dur(member, text) -> seconds. Picks the slowest tempo in range whose bar grid fits the story
    into exactly script.DURATION (shortening elastic gaps by up to 1 - SQUEEZE where that saves a bar),
    then hands the spare bars to the scenes in SLACK_ORDER."""
    total = script.DURATION
    for n_bars in range(int(TEMPO_RANGE[0] * total / 240), int(TEMPO_RANGE[1] * total / 240) + 1):
        beat = total / n_bars / 4
        need = []
        for name, cues in script.SCENES:
            k = 1
            while _fit(name, cues, 0.0, beat, dur, k) is None:
                k += 1
            need.append(k)
        if sum(need) <= n_bars:
            break
    else:
        raise RuntimeError("story does not fit into %.1f s" % total)
    bar = 4 * beat
    spare = n_bars - sum(need)
    order = [i for nm in SLACK_ORDER for i, (n, _) in enumerate(script.SCENES) if n == nm]
    i = 0
    while spare > 0:
        need[order[i % len(order)]] += 1
        spare -= 1
        i += 1
    scenes, lines, marks, sfx = [], [], {}, []
    t_scene, bar0 = 0.0, 0
    for (name, cues), k in zip(script.SCENES, need):
        sq = _fit(name, cues, t_scene, beat, dur, k)
        length0 = _layout_scene(name, cues, t_scene, beat, dur, 0.0, sq)[0]
        slack = max(0.0, k * bar - length0) if any(c[0] == "slack" for c in cues) else 0.0
        while True:
            length, ln, mk, fx = _layout_scene(name, cues, t_scene, beat, dur, slack, sq)
            if length <= k * bar + 1e-6 or slack <= 0:
                break
            slack = max(0.0, slack - (length - k * bar))
        scenes.append(dict(name=name, t0=t_scene, t1=t_scene + k * bar, bars=k, bar0=bar0, squeeze=sq))
        lines += ln
        marks.update(mk)
        sfx += fx
        t_scene += k * bar
        bar0 += k
    return dict(tempo=240.0 * n_bars / total, bar=bar, beat=beat, duration=total, n_bars=n_bars, scenes=scenes,
                lines=lines, marks=marks, sfx=sfx)


def title_letter_times(tl):
    """Landing time of each title glyph (spaces skipped), in 16th notes: 'Pip' on eighths, 'and' and
    'the' as whole words, then 'Paper' and 'Rainbow' letter by letter. Renderer and pops share this."""
    t0 = tl["marks"]["title.letters"]
    s16 = tl["beat"] / 4
    steps = [0, 2, 4, 7, 7, 7, 8, 8, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22]
    return [t0 + k * s16 for k in steps]


class Timeline:
    def __init__(self, d):
        self.d = d
        self.tempo, self.bar, self.beat, self.duration = d["tempo"], d["bar"], d["beat"], d["duration"]
        self.scenes = d["scenes"]
        self.lines = d["lines"]
        self.marks = d["marks"]

    def scene_at(self, t):
        for s in self.scenes:
            if t < s["t1"]:
                return s
        return self.scenes[-1]

    def scene(self, name):
        return next(s for s in self.scenes if s["name"] == name)

    def m(self, key):
        return self.marks[key]

    def lines_in(self, scene):
        return [ln for ln in self.lines if ln["scene"] == scene]

    def line(self, scene, who=None, text_start=None, k=0):
        cand = [ln for ln in self.lines if ln["scene"] == scene and (who is None or ln["who"] == who)
                and (text_start is None or ln["text"].startswith(text_start))]
        return cand[k]


_tl = None


def timeline():
    global _tl
    if _tl is None:
        with open(os.path.join(BUILD, "timeline.json")) as fh:
            _tl = Timeline(json.load(fh))
    return _tl
