"""Render the video.

    python -m mv.render stills 12.0 40.5 ...        # PNG stills into build/stills/
    python -m mv.render video [--from S] [--to S]   # full render -> out/walk_like_an_egyptian.mp4
"""
import argparse
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import skia

from .core import BUILD, FPS, H, ROOT, W, audio

SEG_SECONDS = 8


def render_frame(frame):
    from . import fx, scenes
    t = frame / FPS
    x = scenes.Ctx(t, frame)
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.clear(skia.ColorBLACK)
    post = scenes.SCENE_FN[x.sec](c, x)
    c.resetMatrix()
    if post.get("letterbox"):
        fx.letterbox(c, post["letterbox"])
    fx.bloom(surf, post["bloom"], post["threshold"])
    if post["chroma"] > 0.4:
        fx.chroma(surf, post["chroma"], angle=x.t)
    fx.grade(surf, post["sat"], post["contrast"], post["bright"], post["tint"])
    c = surf.getCanvas()
    if post["scan"] > 0:
        fx.scanlines(c, post["scan"], x.t)
    fx.vignette(c, post["vignette"])
    fx.flash(c, post["flash_col"], post["flash"])
    fx.grain(c, frame, post["grain"])
    if post["fade"] > 0:
        fx.flash(c, "#000000", post["fade"])
    return surf.makeImageSnapshot()


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
        print(f"{path}  {1000 * (time.time() - t0):.0f} ms")


def _encode_segment(args):
    f0, f1, path = args
    if os.path.exists(path):
        return path
    tmp = path + ".part.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
           "-pix_fmt", "yuv420p", "-x264-params", "threads=2", "-g", str(FPS * 2), tmp]
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
    a = audio()
    total = int(math.ceil(a.duration * FPS))
    f_from = int(t_from * FPS)
    f_to = total if t_to is None else min(total, int(t_to * FPS))
    segdir = os.path.join(BUILD, "segments")
    os.makedirs(segdir, exist_ok=True)
    seg = SEG_SECONDS * FPS
    jobs = []
    for s0 in range((f_from // seg) * seg, f_to, seg):
        s1 = min(s0 + seg, total)
        jobs.append((s0, s1, os.path.join(segdir, f"seg_{s0:06d}_{s1:06d}.mp4")))
    t0 = time.time()
    with Pool(workers) as pool:
        paths = pool.map(_encode_segment, jobs, chunksize=1)
    print(f"rendered {len(paths)} segments in {time.time() - t0:.0f}s")
    lst = os.path.join(segdir, "list.txt")
    with open(lst, "w") as fh:
        for p in paths:
            fh.write(f"file '{p}'\n")
    out = out or os.path.join(ROOT, "out", "walk_like_an_egyptian.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    song = os.path.join(BUILD, "song.mp3")
    start = jobs[0][0] / FPS
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
           "-ss", f"{start:.3f}", "-i", song, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
           "-b:a", "256k", "-shortest", "-movflags", "+faststart", out]
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
