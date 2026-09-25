# Procedural animated clips

Fully code-generated animated clips (1920×1080, 30 fps) rendered with
[skia-python](https://github.com/kyamagu/skia-python). Every frame is drawn from scratch — rigged
2D characters, scenery, kinetic typography and a post-processing stack — and everything is driven
by the audio track.

| package | clip |
| --- | --- |
| `paper/` | **Pip and the Paper Rainbow**: a 2:00 paper-craft cartoon for kids. A little paper boat collects six colours from animal friends to make a rainbow. Story, voices, music and sound effects are all generated here |
| `chase/` | **CHASE — 200 Sub Special**: a guitarist roasts the viewer, a pink "cringe" blob steals his trophy, and an all-night chase across a city follows the guitar cover note for note |
| `mv/` | **Walk Like An Egyptian**: a desert-to-dawn dance procession with an Egyptian-themed cast |

The two music videos need songs that are **not** part of this repository. Bring your own copies.
`paper/` needs no outside media: its soundtrack is built from scratch.

## Quick start

```bash
pip install -r requirements.txt   # also needs ffmpeg on PATH and libEGL (apt: libegl1 libgl1)

# CHASE
mkdir -p build_chase && cp /path/to/chase_cover.mp3 build_chase/song.mp3
python -m chase.audio build_chase/song.mp3      # beat grid, envelopes, notes -> build_chase/features.npz
python -m chase.render stills 17.2 70.2 102.5   # preview PNGs -> build_chase/stills/
python -m chase.render video --workers 4        # -> out/chase_200_sub_special.mp4

# Walk Like An Egyptian
mkdir -p build && cp /path/to/song.mp3 build/song.mp3
python -m mv.audio build/song.mp3
python -m mv.render video --workers 4           # -> out/walk_like_an_egyptian.mp4
```

Rendered segments are cached (`build*/segments/`); delete one to re-render just that stretch.
`--from/--to` limit a render to a time range.

## Pip and the Paper Rainbow

A two-minute story for small children in a cut-paper look: torn-edge clouds, hills and water strips
with white fibrous rims, scissor-cut puppets on gold split pins, a shared paper-stock texture on
every piece, soft cast shadows between layers, blurred foreground pieces for depth of field and
sticker-style cut-paper letters.

```bash
python -m venv .venv-paper && . .venv-paper/bin/activate   # kokoro-onnx wants numpy 2
pip install -r paper/requirements.txt
sudo apt install ffmpeg fluidsynth musescore-general-soundfont libegl1 libgl1
python -m paper.audio              # voices, score, mix, timeline -> build_paper/ (downloads the TTS model once)
python -m paper.audio --check      # optional: Whisper round trip of every line (needs sherpa-onnx + model, see below)
python -m paper.render stills 30 116
python -m paper.render video --workers 4        # -> out/pip_and_the_paper_rainbow.mp4
```

### Story (`paper/script.py`)

Grey sky after the rain. Pip, a boat folded from notebook paper, wants to make a rainbow, and a
rainbow needs six colours. Each friend hands over a paper streamer that ties onto Pip's toothpick
mast; a tracker badge fills in arc by arc and each colour word pops up in cut-paper letters. At the
end the streamers fly up, one arc per beat, the clouds clear and everyone cheers.

| scene | time | friend / event |
| --- | --- | --- |
| title | 0–7.2 s | title letters land on the beat; the card slides away like a sheet of paper |
| garden | 7.2–24.0 | raindrops on threads are pulled up, "the sky is so grey", idea bulb, the six-colour tracker |
| red | 24.0–36.0 | Lulu the Ladybug flies the red streamer over |
| orange | 36.0–48.0 | Finn the goldfish leaps and flicks it from his tail |
| yellow | 48.0–62.4 | three ducklings: "Yes, yes!" "Yes!" "Yes, of course!" |
| green | 62.4–74.4 | Freddy the Frog hops pad to pad and delivers with his tongue |
| blue | 74.4–88.8 | a whale rises from the sea; the streamer rides up the spout |
| purple | 88.8–100.8 | Olive the Octopus waves all eight arms and reaches over |
| finale | 100.8–120 | "Red! Orange! Yellow! Green! Blue! Purple!" — one arc per beat, blue sky, confetti, The End |

Scene changes are foreground wipes: a torn sheet of grass, leaves, reeds or waves sweeping past
the lens.

### Audio (`paper/audio.py`, `voice.py`, `timeline.py`, `music.py`, `sfx.py`)

* **Voices** — [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) through
  [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), one voice per character. Critters are
  pitched up by resampling (formants move too, so they sound small), the whale is pitched down; Kokoro
  renders each line slower by the same ratio so the speaking rate stays natural. Every line was run back
  through Whisper speech recognition, and lines it misheard were reworded ("Hi, Finn" came back as
  "Python", so Pip says "Hello, Finn").
* **Timeline** — the cue sheet is laid out on a constant-tempo bar grid: every scene is a whole number
  of bars, elastic pauses shrink a little where that saves a bar, and the tempo is chosen so the film is
  exactly 2:00 (100 BPM, 50 bars). Marks in the cue sheet drive both the animation and the effects.
* **Music** — an original underscore written as MIDI and played through FluidSynth with the MuseScore
  General soundfont: each friend has an instrument (glockenspiel, marimba, oboe and bassoon, tuba and
  woodblocks, horn and harp, xylophone), the melody thins out under speech, every colour word lands on a
  V–I cadence with a glockenspiel "ta-da", and the finale climbs one scale step per rainbow arc.
* **Effects** — synthesised with numpy: paper swishes and crackle, drips, splashes, boing, peeps,
  bubbles, whale call, spout, sparkles, confetti, plus stream/pond/sea/bird ambiences.
* **Mix** — music ducked 11 dB under speech with an extra cut in the speech band, loudness-normalised to
  −16 LUFS with a look-ahead limiter. Per-line voice envelopes drive the lip-sync.

For `--check`, put [sherpa-onnx-whisper-base.en](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
in `build_paper/models/`.

### Modules

| module | role |
| --- | --- |
| `paper/craft.py` | paper stock texture, scissor-cut and torn outlines, cast shadows, split pins, cut-paper letters, parallax camera, baked layers, light/vignette/grain |
| `paper/cast.py` | Pip, Lulu, Finn, the ducklings, Freddy, the whale and Olive as paper puppets with shared faces |
| `paper/world.py` | skies, torn clouds and hills, water strips, trees, flowers, reeds, lily pads, island, lighthouse, sun, rainbow, tracker badge, splashes, confetti |
| `paper/anim.py` | lip-sync, blinks, streamer hand-offs, the tracker, colour words |
| `paper/scenes.py`, `paper/transitions.py` | the nine scenes, the sheet slide and the foreground wipes |
| `paper/render.py` | multiprocess segment renderer + audio mux |

## CHASE

### Audio analysis (`chase/audio.py`)

* **Beat grid** — comb-filter search over the percussive onset envelope: 112.1 BPM, first downbeat
  (the band's entry) at 24.21 s; chord-change novelty confirms the bar phase. Beats are numbered from
  that downbeat, negative before it.
* **Notes** — every guitar note transcribed with Spotify's
  [basic-pitch](https://github.com/spotify/basic-pitch) (1,174 notes) plus a "lead" top-voice subset.
  They drive the fret hand (it slides to the played pitch), string glow, sparks, the solo's lightning
  bolts, and the "melody ribbon" — a glowing line that traces the guitar line across the sky.
* **Talking intro** — speech-band envelope for lip-sync and the seven phrase boundaries the speech
  bubbles are timed to.
* **Whammy dive** — pYIN pitch curve of the dive-bomb (G#2 → B1 → back) and the slide into the band's
  entry; the "power line" and the whole room sag with it.
* Kick / snare / hat / bass / level envelopes at 100 Hz for camera punches, flashes and strobes.

### On-screen text (`chase/lyrics.py`)

All words on screen are **original lines written for this clip** (not the song's lyrics): Zip's
speech bubbles, a fake live-chat, neon billboards, train LED boards, tunnel graffiti, chorus slams and
manga captions. Each line sits on a bar window and its syllables are snapped to real transcribed note
onsets, so the text "sings along" with the melody the guitar plays.

### Modules

| module | role |
| --- | --- |
| `chase/hero.py` | ZIP, a 2.5D cel-shaded puppet that turns from front to profile: IK legs with knees that always bend toward the facing direction, IK arms, anime face with expression presets and lip-sync, spiky hair and scarf with secondary motion, a playable guitar |
| `chase/cast.py` | the wobbling pink blob (moods, pseudopods), AMPY the amp-bot, the 200 trophy, crowds |
| `chase/world.py` | practice room + webcam UI, pre-rendered parallax city strips, rooftops with gaps placed where Zip leaps, train with LED boards, tunnel, radio tower, the "cringe dimension" |
| `chase/fxk.py` | lightning, manga speed lines, halftone screentone, SFX lettering, shockwaves, impact frames, glitch, zoom blur |
| `chase/timeline.py` | beat context, speech bubbles, text renderers (slam / mega / fly / float / sky), melody ribbon |
| `chase/scenes_[abc].py` | the scenes |
| `chase/render.py` | multiprocess segment renderer + audio mux |

### Song map (bar 0 = 24.21 s)

| bars | time | music | scene |
| --- | --- | --- | --- |
| – | 0–24.2 s | spoken intro, chord, whammy dive, slide | webcam roast: speech bubbles + live chat; the word CRINGE falls out of the bubble and becomes the blob; KRANG chord; the power line sags with the dive; SKRRRT slash |
| 0 | 24.2 | band enters | CHASE title slams letter-per-eighth; the blob escapes with the trophy |
| 1–8 | 26.4–43.5 | intro riff / verse | rooftop chase (leaps land on bar lines, neon billboards light word by word), then "blob-cam" with words flying at the camera |
| 9–16 | 43.5–60.6 | verse 2 | surfing the train roof while playing (LED boards), strobing tunnel graffiti |
| 17–20 | 60.6–69.2 | pre-chorus + stop | the track runs out; tension cuts and a 3-2-1 count; freeze against the moon on the stop |
| 21–27 | 69.2–84.2 | chorus 1 | lightning-surfing over the city: side, hero front shot, aerial with the melody as the lightning trail |
| 28–31 | 84.2–92.7 | post-chorus + stop | manga panels: ZAP / SPLAT / SPLIT, mini-blobs, BONK, they merge into a giant |
| 32–39 | 92.7–109.9 | guitar solo | kaiju showdown from a radio tower: every solo note fires a bolt, the melody ribbon streams out of the guitar |
| 40–44 | 109.9–120.6 | pre-chorus 2 + stop | swallowed into the cringe dimension; the story's turn ("everybody starts out cringe"); light cracks through |
| 45–58 | 120.6–150.5 | final chorus | dawn breakout, sky dive after the trophy, fireworks over cheering rooftops, slow-motion reach, GOT IT, hero landing, friends arrive |
| 59–62 | 150.5–159.1 | outro riff | victory jam at sunrise with AMPY and the (now friendly) blob |
| 63– | 159.1–end | final chord | freeze-frame, then back to the webcam sign-off and end card |

## Walk Like An Egyptian

| module | role |
| --- | --- |
| `mv/audio.py` | beat grid (~103 BPM), kick/snare/hat envelopes, mel spectrum |
| `mv/rig.py` | Egyptian-style dancers (IK legs, choreography styles, flat / silhouette / neon / tomb-paint styles) |
| `mv/props.py` | skies, pyramids, dunes, palms, sphinx, boats, symbols, particles |
| `mv/scenes.py` | 15 scenes mapped to the song's sections |
| `mv/fx.py`, `mv/render.py` | post effects, renderer |

## Credits

The paper clip's voices come from Kokoro-82M (Apache 2.0) and its instruments from the MuseScore General
soundfont (MIT); both are downloaded or installed at build time, not stored here.

## Fonts

`assets/fonts/` contains fonts from Google Fonts under the SIL Open Font License: Bangers, Black Ops
One, Bungee, Cinzel / Cinzel Decorative, Dela Gothic One, Luckiest Guy, Monoton, Noto Sans Egyptian
Hieroglyphs, Patrick Hand, Permanent Marker, Press Start 2P, Rampart One, Rubik Glitch, Rubik Mono One,
Titan One, VT323.
