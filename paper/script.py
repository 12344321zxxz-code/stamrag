"""The story, the cast's voices and the cue sheet for PIP AND THE PAPER RAINBOW.

Every scene is a list of cues that the audio build lays out in time:

    ("say", who, text)            a spoken line (who may be a list: they speak together)
    ("beats", who, words, n)      one word every n beats, starting on the current beat
    ("gap", seconds)              a pause (elastic: the layout may shorten it to fit the bar grid)
    ("hold", seconds)             a pause that is never shortened
    ("wait_beats", n)             a pause of n beats
    ("align", "beat" | "bar")     wait for the next beat / bar line of the score
    ("mark", name)                a named moment the animation (and the sound effects) key off
    ("sfx", name, dt, {kw})       a sound effect dt seconds after the current time (time does not advance)

A scene lasts a whole number of bars; the tempo is chosen so that the film is exactly DURATION long.
"""

TITLE = "Pip and the Paper Rainbow"
DURATION = 120.0

# who: (kokoro voice, pitch shift in semitones, speaking rate after the shift, language)
VOICES = {
    "narrator": ("af_heart", 0, 0.88, "en-us"),
    "pip": ("af_bella", 4, 0.95, "en-us"),
    "lulu": ("bf_lily", 6, 0.95, "en-gb"),
    "finn": ("am_puck", 3, 1.0, "en-us"),
    "duck1": ("af_nova", 7, 1.0, "en-us"),
    "duck2": ("af_sky", 8, 1.0, "en-us"),
    "duck3": ("bf_isabella", 7, 1.0, "en-gb"),
    "freddy": ("am_michael", -1, 0.95, "en-us"),
    "whale": ("bm_george", -4, 0.9, "en-gb"),
    "olive": ("bf_emma", 2, 0.95, "en-gb"),
}
GROUPS = {
    "friends": ["finn", "freddy", "olive", "lulu", "whale"],
}

COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]


def _color_scene(name, color, reveal, friend, ask, answer, give_gap=1.35, lead=1.0, extra=(), give=()):
    return (name, [
        ("gap", lead),
        ("mark", "reveal"),
        *extra,
        ("say", "narrator", reveal),
        ("gap", 0.3),
        ("mark", "ask"),
        ("say", "pip", ask),
        ("gap", 0.25),
        ("mark", "answer"),
        *([("say", friend, answer)] if isinstance(answer, str) else
          [c for k, (who, txt) in enumerate(answer) for c in ([("gap", 0.12)] if k else []) + [("say", who, txt)]]),
        ("gap", 0.15),
        ("mark", "give"),
        *give,
        ("sfx", "whoosh_up", 0.1, {}),
        ("sfx", "sparkle", give_gap - 0.25, {}),
        ("hold", give_gap),
        ("align", "beat"),
        ("mark", "word"),
        ("say", "narrator", color.capitalize() + "!"),
        ("gap", 1.1),
    ])


SCENES = [
    ("title", [
        ("gap", 0.25),
        ("mark", "letters"),
        ("wait_beats", 6),
        ("say", "narrator", "Pip! And the Paper Rainbow!"),
        ("gap", 0.7),
    ]),
    ("garden", [
        ("gap", 0.5),
        ("mark", "drips"),
        ("sfx", "drip", 0.0, {"pitch": 1.0}),
        ("sfx", "drip", 0.42, {"pitch": 1.25}),
        ("hold", 0.7),
        ("say", "narrator", "Drip, drop! The rain has stopped."),
        ("mark", "rain_up"),
        ("gap", 0.35),
        ("say", "narrator", "In a garden puddle floats Pip, a little paper boat."),
        ("gap", 0.35),
        ("mark", "hello"),
        ("say", "pip", "Hello! Oh, the sky is so grey."),
        ("gap", 0.3),
        ("mark", "idea"),
        ("sfx", "idea", 0.0, {}),
        ("say", "pip", "I know! Let's make a rainbow!"),
        ("gap", 0.35),
        ("mark", "tracker"),
        ("say", "narrator", "A rainbow needs six colors. Let's go and find them!"),
        ("mark", "sail"),
        ("sfx", "paddle", 0.0, {}),
        ("hold", 1.1),
    ]),
    _color_scene("red", "red", "Look! It's Lulu the Ladybug.", "lulu",
                 "Hi, Lulu! May I have some red?", "Of course, Pip! Here you go!",
                 extra=[("sfx", "pop", 0.05, {"pitch": 1.3})], give=[("sfx", "flutter", 0.0, {})]),
    _color_scene("orange", "orange", "Splish, splash! Here comes Finn the Fish.", "finn",
                 "Hello, Finn! May I have some orange?", "Catch, Pip! Here it comes!",
                 extra=[("sfx", "splash", 0.0, {"size": 0.6}), ("sfx", "splash", 0.55, {"size": 0.8})],
                 give=[("sfx", "splash", 0.05, {"size": 0.5}), ("sfx", "splash", 1.0, {"size": 0.7})]),
    _color_scene("yellow", "yellow", "Quack, quack! Three little ducklings paddle by.", "ducks",
                 "Hi, ducklings! May I have some yellow?",
                 [("duck1", "Yes, yes!"), ("duck2", "Yes!"), ("duck3", "Yes, of course!")],
                 extra=[("sfx", "peeps", 0.0, {})], give=[("sfx", "peeps", 0.0, {"n": 5})]),
    _color_scene("green", "green", "Boing! Freddy the Frog hops onto a lily pad.", "freddy",
                 "Hi, Freddy! May I have some green?", "Ribbit! Green is for you!",
                 extra=[("sfx", "boing", 0.0, {})], give=[("sfx", "tongue", 0.15, {})]),
    _color_scene("blue", "blue", "Whoosh! Out at sea, a big whale says hello.", "whale",
                 "Hello, Whale! May I have some blue?", "Hellooo, little Pip. Here is my blue.",
                 give_gap=1.6, lead=1.2,
                 extra=[("sfx", "whale_rise", 0.0, {}), ("sfx", "whale_call", 0.9, {})],
                 give=[("sfx", "spout", 0.0, {})]),
    _color_scene("purple", "purple", "Last of all, Olive the Octopus waves all eight arms!", "olive",
                 "Hello, Olive! May I have some purple?", "Purple for Pip! Here you go!",
                 extra=[("sfx", "bubbles", 0.0, {"n": 9})], give=[("sfx", "bubbles", 0.0, {"n": 5})]),
    ("finale", [
        ("gap", 0.8),
        ("say", "narrator", "Now Pip has all six colors!"),
        ("gap", 0.3),
        ("mark", "up"),
        ("say", "pip", "Now, up, up, up you go!"),
        ("mark", "launch"),
        ("sfx", "whoosh_up", 0.0, {"d": 1.2}),
        ("gap", 0.4),
        ("align", "bar"),
        ("mark", "arcs"),
        ("beats", "narrator", [c.capitalize() + "!" for c in COLORS], 2),
        ("align", "bar"),
        ("mark", "hooray"),
        ("sfx", "confetti", 0.0, {}),
        ("say", "friends", "Hurray!"),
        ("gap", 0.15),
        ("mark", "look"),
        ("say", "pip", "Look, a rainbow!"),
        ("gap", 0.45),
        ("mark", "thanks"),
        ("say", "pip", "Thank you, friends!"),
        ("gap", 0.5),
        ("slack",),
        ("align", "bar"),
        ("mark", "end"),
        ("say", "narrator", "The End."),
        ("hold", 1.1),
    ]),
]
