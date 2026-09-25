"""The clip's on-screen text: ORIGINAL lines written for this video (not the song's lyrics).

Each line is placed on a bar window; its syllables are snapped to real guitar-note onsets
transcribed from the track, so the words 'sing along' with the melody the guitar plays.
"""
import math
import re

import numpy as np

from .core import audio, clamp, ease_back, ease_out

A = audio()

# (first bar, bars, text, style)
LINES = [
    (1, 2, "GET BACK HERE, YOU PINK DISASTER", "neon"),
    (3, 2, "THAT TROPHY TOOK TWO HUNDRED FRIENDS", "neon"),
    (5, 2, "YOU CAN BOUNCE, BUT YOU CAN'T HIDE", "fly"),
    (7, 2, "SIX STRINGS AND ZERO BRAKES", "fly"),
    (9, 2, "NEXT STOP: YOUR TOTAL DEFEAT", "led"),
    (11, 2, "MIND THE GAP AND MIND THE RIFF", "led"),
    (13, 2, "NO SIGNAL DOWN HERE, JUST NOISE", "graffiti"),
    (15, 2, "EYES ON THE PRIZE, HANDS ON THE FRETS", "graffiti"),
    (17, 2, "THE TRACK RUNS OUT... I DON'T", "slam"),
    (21, 2, "CHASE IT DOWN!", "mega"),
    (23, 2, "LIGHTNING IN MY HANDS, THUNDER IN THE AMP", "slam"),
    (25, 2, "THE WHOLE CITY'S WIDE AWAKE", "slam"),
    (27, 1, "CHASE IT DOWN AT FULL VOLUME", "slam"),
    (40, 2, "IT'S PINK. IT'S STICKY. IT'S SO CRINGE.", "float"),
    (42, 1, "WAIT... IS THAT MY FIRST VIDEO?", "float"),
    (43, 1, "EVERYBODY STARTS OUT CRINGE.", "float"),
    (44, 0.8, "...THAT'S HOW YOU GET GOOD.", "float"),
    (45, 2, "CHASE IT DOWN!", "mega"),
    (47, 2, "EVERY CRINGE BEGINNING IS A STARTING LINE", "slam"),
    (49, 2, "TWO HUNDRED STRONG AND JUST WARMING UP", "slam"),
    (51, 2, "LOUDER! FASTER! HIGHER! GO!", "slam"),
    (53, 2, "REACH OUT... ALMOST... THERE...", "slam"),
    (55, 1, "GOT IT!!!", "mega"),
    (56, 3, "CHASE IT DOWN, ALL THE WAY TO 2,000", "slam"),
    (59, 2, "THANKS FOR 200", "sky"),
    (61, 2, "SEE YOU AT THE NEXT ONE", "sky"),
]

SPECIAL_SYL = {"200": 3, "2,000": 3, "2000": 3, "CRINGE": 1, "FRIENDS": 1, "STRINGS": 1, "CITY'S": 2, "VIDEO": 3,
               "EVERYBODY": 4, "DON'T": 1, "CAN'T": 1, "IT'S": 1, "THAT'S": 1, "BEGINNING": 3, "HIGHER": 2,
               "LIGHTNING": 2, "FIRE": 1, "HUNDRED": 2}


def syllables(word):
    w = re.sub(r"[^A-Z0-9,']", "", word.upper())
    if not w:
        return 0
    if w in SPECIAL_SYL:
        return SPECIAL_SYL[w]
    w2 = re.sub(r"[^A-Z]", "", w)
    if not w2:
        return 1
    groups = re.findall(r"[AEIOUY]+", w2)
    n = len(groups)
    if w2.endswith("E") and n > 1 and not w2.endswith("LE"):
        n -= 1
    return max(1, n)


def _onsets(t0, t1):
    st = A.notes[:, 0]
    amp = A.notes[:, 3]
    m = (st >= t0) & (st < t1)
    cand = sorted(zip(np.round(st[m], 3), amp[m]))
    merged = []
    for o, a in cand:
        if merged and o - merged[-1][0] < 0.08:
            if a > merged[-1][1]:
                merged[-1] = (merged[-1][0], a)
            continue
        merged.append((o, a))
    return [o for o, _ in merged]


class Line:
    def __init__(self, bar, bars, text, style):
        self.bar, self.bars, self.text, self.style = bar, bars, text, style
        self.t0 = A.bar_time(bar)
        self.t1 = A.bar_time(bar + bars)
        self.words = text.split(" ")
        syl = [syllables(w) for w in self.words]
        n = max(1, sum(syl))
        on = _onsets(self.t0 - 0.03, self.t1 - 0.3)
        if len(on) >= n:
            # spread the syllables over the leading part of the window's note onsets (keeps order & uniqueness)
            span = min(len(on) - 1, int(round((len(on) - 1) * min(1.0, 0.45 + n / len(on)))))
            k = np.linspace(0, span, n) if n > 1 else [0]
            times = [on[int(round(v))] for v in k]
        else:
            grid = [A.beat_time(bar * 4 + j * 0.5) for j in range(int(bars * 8))]
            times = sorted(set(on) | set(grid[: max(0, n - len(on))]))[:n]
            while len(times) < n:
                times.append(times[-1] + A.period * 0.5)
        self.word_t = []
        j = 0
        for w, s in zip(self.words, syl):
            self.word_t.append(times[min(j, len(times) - 1)] if s > 0 else (self.word_t[-1] if self.word_t else times[0]))
            j += max(s, 0)
        self.end_t = self.t1

    def state(self, t, fade_in=0.25, fade_out=0.35):
        """Returns (alpha, [(word, reveal 0..1+, is_current, age)])."""
        if t < self.t0 - fade_in or t > self.end_t + fade_out:
            return 0.0, []
        a = clamp((t - (self.t0 - fade_in)) / fade_in) * clamp((self.end_t + fade_out - t) / fade_out)
        out = []
        cur = -1
        for i, wt in enumerate(self.word_t):
            if t >= wt:
                cur = i
        for i, (w, wt) in enumerate(zip(self.words, self.word_t)):
            age = t - wt
            out.append((w, age, i == cur))
        return a, out


_lines = None


def lines():
    global _lines
    if _lines is None:
        _lines = [Line(*l) for l in LINES]
    return _lines


def active(t, styles=None):
    return [l for l in lines() if (styles is None or l.style in styles) and l.t0 - 0.3 <= t <= l.end_t + 0.4]


# ----------------------------------------------------------------------------
# cold-open dialogue (original lines) timed to the spoken phrases in the audio
# ----------------------------------------------------------------------------
INTRO_BUBBLES = [
    "...hm?",
    "Oh. It's you.",
    "A 200 sub special?",
    "You actually clicked on this?",
    "Honestly...",
    "...that's kinda CRINGE.",
    "Whatever. Watch and learn.",
]

CHAT = [  # (time, user, message, colour)
    (1.2, "zip_fan99", "FIRST", "#ffd21f"),
    (2.3, "voltkid", "200 LETS GOOO", "#2fe3d7"),
    (4.0, "moth_lamp", "he looks annoyed lol", "#ff9ad0"),
    (6.6, "bytebass", "200 sub special!!", "#9aff7a"),
    (8.6, "cheesecat", "yes we clicked", "#ffb35a"),
    (12.6, "moth_lamp", "did he just call us cringe", "#ff9ad0"),
    (13.6, "voltkid", "WHAT is that pink thing", "#2fe3d7"),
    (15.3, "bytebass", "behind you!!!", "#9aff7a"),
    (17.4, "zip_fan99", "PLAY THE THING", "#ffd21f"),
    (20.2, "cheesecat", "the whammy omg", "#ffb35a"),
    (22.4, "voltkid", "here it comes", "#2fe3d7"),
]

OUTRO_BUBBLES = [(162.6, 165.2, "...200 subs, huh. Not bad."), (165.4, 168.6, "Still cringe though.")]
