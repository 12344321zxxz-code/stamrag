# Walk Like An Egyptian — procedural music video

A fully code-generated, beat-synced animated clip (1920×1080, 30 fps) rendered with
[skia-python](https://github.com/kyamagu/skia-python). Every frame is drawn from scratch: an
original cast of rigged 2D dancers, hand-built props and scenery, and a post-processing stack
(bloom, chromatic aberration, colour grade, grain, vignette). All of it is driven by an
analysis of the audio track.

The song itself is **not** part of this repository. Bring your own copy.

## Quick start

```bash
pip install -r requirements.txt          # also needs ffmpeg on PATH and libEGL (apt: libegl1 libgl1)
mkdir -p build && cp /path/to/song.mp3 build/song.mp3
python -m mv.audio build/song.mp3        # beat grid + envelopes -> build/features.npz
python -m mv.render stills 17 38 92      # preview PNGs -> build/stills/
python -m mv.render video --workers 4    # full render -> out/walk_like_an_egyptian.mp4
```

Segments are cached in `build/segments/` (8 s each). Delete one to re-render just that part.
`--from/--to` limit a render to a time range.

## How it works

| module | role |
| --- | --- |
| `mv/audio.py` | librosa beat tracking, regularised onto a locally-corrected constant-tempo grid (~103 BPM); kick/snare/hat flux envelopes, RMS, 32-band mel spectrum at 100 Hz |
| `mv/core.py` | colours, paint helpers, easing, noise, and the `Audio` timing object (`beat(t)`, `bar_time(n)`, envelopes) |
| `mv/rig.py` | the dancers: FK arms, 2-bone IK legs whose feet plant in world space on every beat, choreography styles (`egypt_walk`, `toggle`, `hook`, `praise`, `moonwalk`, …), and flat / silhouette / neon / tomb-paint render styles |
| `mv/props.py` | skies, stars, sun (with Amarna-style hand rays), pyramids, dunes, palms, obelisks, sphinx, felucca, solar barque, lotus, Eye of Horus, ankh, scarabs, columns, torches, fireworks, confetti, synthwave grid |
| `mv/scenes.py` | the timeline: one scene per song section, each with several shots cut on bar lines |
| `mv/fx.py` | full-frame post effects |
| `mv/render.py` | multiprocess segment renderer piping raw frames to x264, then concat + audio mux |

### Cast (original designs)

Nefi (bob wig, linen dress), Khet and Tut (striped headcloths), Anpu (jackal-headed),
Miu (cat-headed), Heru (falcon-headed with sun disk), Djehu (ibis-headed), Sobi (crocodile-headed),
Meri (red dress) and a moonwalking mummy.

### Song map (bars from the detected grid, beat 0 = first downbeat at 0.88 s)

| bars | time | section | scene |
| --- | --- | --- | --- |
| –2 | 0–5.5 s | intro | sunrise between the pyramids, title lands word-by-word on the beat, dive into the sun |
| 2–15 | 5.5–35.9 | verse 1 | golden-desert procession: wide, close tracking, silhouettes against a giant sun |
| 15–18 | 35.9–42.9 | hook 1 | a tomb wall's painted registers come alive |
| 18–20 | 42.9–47.5 | break | flight down a torch-lit pyramid corridor |
| 20–33 | 47.5–77.8 | verse 2 | night on the Nile: boat party, close-up, underwater croc walk |
| 33–36 | 77.8–84.8 | hook 2 | sphinx laser stage with crowd |
| 36–38 | 84.8–89.5 | break | scarab swarm assembles the Eye of Horus, then bursts |
| 38–50 | 89.5–117.4 | verse 3 | synthwave Egypt, 2×2 and 3×3 dance-off panels, mummy moonwalk solo |
| 50–54 | 117.4–126.7 | bridge | the night sky draws a dancing constellation |
| 54–67 | 126.7–157.0 | verse 4 | solar barque at dawn, then the full parade |
| 67–70 | 157.0–164.0 | hook 3 | eight-way kaleidoscope |
| 70–73 | 164.0–171.0 | break | shadow play in the hall of columns |
| 73 | 171.0–173.3 | stop | the near-silent bar: frozen spotlight, the eye opens |
| 74–81 | 173.3–189.6 | finale | fireworks and searchlights, rapid close-up montage, everyone together |
| 81– | 189.6–end | outro | the troupe walks into the sunset, end title |

## Fonts

`assets/fonts/` contains fonts from Google Fonts under the SIL Open Font License:
Cinzel / Cinzel Decorative, Noto Sans Egyptian Hieroglyphs, Bungee, Monoton, Titan One.
