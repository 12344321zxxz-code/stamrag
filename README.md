# Procedural music videos

Two fully code-generated, beat-synced animated clips (1920×1080, 30 fps) rendered with
[skia-python](https://github.com/kyamagu/skia-python). Every frame is drawn from scratch — rigged
2D characters, scenery, kinetic typography and a post-processing stack — and everything is driven
by an analysis of the audio track.

| package | clip |
| --- | --- |
| `chase/` | **CHASE — 200 Sub Special**: a guitarist roasts the viewer, a pink "cringe" blob steals his trophy, and an all-night chase across a city follows the guitar cover note for note |
| `mv/` | **Walk Like An Egyptian**: a desert-to-dawn dance procession with an Egyptian-themed cast |

The songs themselves are **not** part of this repository. Bring your own copies.

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

## Fonts

`assets/fonts/` contains fonts from Google Fonts under the SIL Open Font License: Bangers, Black Ops
One, Bungee, Cinzel / Cinzel Decorative, Dela Gothic One, Luckiest Guy, Monoton, Noto Sans Egyptian
Hieroglyphs, Patrick Hand, Permanent Marker, Press Start 2P, Rampart One, Rubik Glitch, Rubik Mono One,
Titan One, VT323.
