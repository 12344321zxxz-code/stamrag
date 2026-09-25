"""Render PIP AND THE PAPER RAINBOW.

    python -m paper.audio                              # first: voices, score, mix, timeline
    python -m paper.render stills 3.5 12 40.2 ...      # PNG stills -> build_paper/stills/
    python -m paper.render video [--from S] [--to S]   # -> out/pip_and_the_paper_rainbow.mp4
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import skia

from .core import BUILD, FPS, H, ROOT, W

SEG_SECONDS = 5
WIPE_HALF = 0.45
SLIDE = (0.35, 0.6)
WIPES = dict(red="grass", orange="leaves", yellow="reeds", green="leaves2", blue="wave", purple="wave2",
             finale="wave3")


def _scenes():
    from . import scenes as S
    return {name: getattr(S, "s_" + name) for name in
            ("title", "garden", "red", "orange", "yellow", "green", "blue", "purple", "finale")}


def draw_scene(c, name, t):
    c.save()
    _scenes()[name](c, t)
    c.restore()


def render_frame(frame):
    from .craft import finish
    from .timeline import timeline
    from . import transitions as X
    tl = timeline()
    t = frame / FPS
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.clear(skia.ColorWHITE)
    scenes = tl.scenes
    cur = tl.scene_at(t)
    idx = scenes.index(cur)
    nxt = scenes[idx + 1] if idx + 1 < len(scenes) else None
    prv = scenes[idx - 1] if idx > 0 else None
    # title card slides away like a sheet of paper
    if cur["name"] == "garden" and t < cur["t0"] + SLIDE[1] or cur["name"] == "title" and nxt and \
            t > nxt["t0"] - SLIDE[0]:
        T = scenes[1]["t0"]
        draw_scene(c, "garden", t)
        X.sheet_slide(c, lambda cc: draw_scene(cc, "title", t), clamp01((t - (T - SLIDE[0])) / sum(SLIDE)))
    else:
        draw_scene(c, cur["name"], t)
        for a, b in ((prv, cur), (cur, nxt)):
            if a is None or b is None or b["name"] not in WIPES:
                continue
            T = b["t0"]
            if abs(t - T) < WIPE_HALF:
                X.wipe(c, WIPES[b["name"]], (t - (T - WIPE_HALF)) / (2 * WIPE_HALF))
    finish(c, frame)
    fade = clamp01((t - (tl.duration - 0.8)) / 0.75)
    if fade > 0:
        c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Color=skia.Color(0, 0, 0, int(255 * fade))))
    return surf.makeImageSnapshot()


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def frame_bytes(frame):
    return render_frame(frame).tobytes()


def stills(times, outdir):
    os.makedirs(outdir, exist_ok=True)
    for s in times:
        f = int(round(float(s) * FPS))
        t0 = time.time()
        img = render_frame(f)
        path = os.path.join(outdir, f"still_{float(s):07.2f}.png")
        img.save(path)
        print(f"{path}  {1000 * (time.time() - t0):.0f} ms", flush=True)


def _encode_segment(args):
    f0, f1, path = args
    if os.path.exists(path):
        return path
    tmp = path + ".part.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-tune", "animation", "-pix_fmt", "yuv420p", "-x264-params", "threads=2", "-g", str(FPS * 2), tmp]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f0, f1):
        proc.stdin.write(frame_bytes(f))
    proc.stdin.close()
    proc.wait()
    os.replace(tmp, path)
    print(f"segment {os.path.basename(path)} frames {f0}-{f1} in {time.time() - t0:.0f}s", flush=True)
    return path


def video(t_from=0.0, t_to=None, workers=4, out=None):
    from .timeline import timeline
    tl = timeline()
    total = int(round(tl.duration * FPS))
    f_from = int(t_from * FPS)
    f_to = total if t_to is None else min(total, int(t_to * FPS))
    segdir = os.path.join(BUILD, "segments")
    os.makedirs(segdir, exist_ok=True)
    seg = SEG_SECONDS * FPS
    jobs = []
    for s0 in range((f_from // seg) * seg, f_to, seg):
        s1 = min(s0 + seg, total)
        jobs.append((s0, s1, os.path.join(segdir, f"seg_{s0:05d}_{s1:05d}.mp4")))
    t0 = time.time()
    with Pool(workers) as pool:
        paths = pool.map(_encode_segment, jobs, chunksize=1)
    print(f"rendered {len(paths)} segments in {time.time() - t0:.0f}s")
    lst = os.path.join(segdir, "list.txt")
    with open(lst, "w") as fh:
        for p in paths:
            fh.write(f"file '{p}'\n")
    out = out or os.path.join(ROOT, "out", "pip_and_the_paper_rainbow.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    start = jobs[0][0] / FPS
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
           "-ss", f"{start:.3f}", "-i", os.path.join(BUILD, "soundtrack.wav"), "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stills", "video"])
    ap.add_argument("times", nargs="*")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.mode == "stills":
        stills(a.times, os.path.join(BUILD, "stills"))
    else:
        video(a.t_from, a.t_to, a.workers, a.out)


if __name__ == "__main__":
    sys.exit(main())
