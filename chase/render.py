"""Render the CHASE clip.

    python -m chase.render stills 12.0 40.5 ...        # PNG stills -> build_chase/stills/
    python -m chase.render video [--from S] [--to S]   # full render -> out/chase_200_sub_special.mp4
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

SEG_SECONDS = 6


def _scene_fn(name):
    from . import scenes_a, scenes_b, scenes_c
    return {
        "cold": scenes_a.s_cold, "title": scenes_a.s_title, "roof": scenes_a.s_roof,
        "train": scenes_b.s_train, "build": scenes_b.s_build, "chorus1": scenes_b.s_chorus1,
        "panels": scenes_b.s_panels, "solo": scenes_c.s_solo, "inside": scenes_c.s_inside,
        "final": scenes_c.s_final, "victory": scenes_c.s_victory, "end": scenes_c.s_end,
    }[name]


def render_frame(frame):
    from . import fxk as FX
    from .timeline import Ctx
    t = frame / FPS
    x = Ctx(t, frame)
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.clear(skia.ColorBLACK)
    pp = _scene_fn(x.sec)(c, x)
    c.resetMatrix()
    if pp.get("zoom_blur", 0) > 0.004:
        FX.zoom_blur(surf, pp["zoom_blur"], *pp["zb_center"])
    sm = pp.get("smear", (0, 0))
    if abs(sm[0]) + abs(sm[1]) > 2:
        FX.motion_smear(surf, sm[0], sm[1])
    FX.bloom(surf, pp["bloom"], pp["threshold"])
    if pp["chroma"] > 0.4:
        FX.chroma(surf, pp["chroma"], angle=t * 1.7)
    FX.grade(surf, pp["sat"], pp["contrast"], pp["bright"], pp["tint"])
    if pp.get("glitch", 0) > 0.01:
        FX.glitch(surf, pp["glitch"], seed=frame)
    if pp.get("impact", 0) > 0:
        FX.impact(surf, pp["impact"], invert=pp.get("invert", True))
    c = surf.getCanvas()
    if pp.get("letterbox"):
        FX.letterbox(c, pp["letterbox"])
    if pp["scan"] > 0:
        FX.scanlines(c, pp["scan"], t)
    FX.vignette(c, pp["vignette"])
    FX.flash(c, pp["flash_col"], pp["flash"])
    FX.grain(c, frame, pp["grain"])
    if pp["fade"] > 0:
        FX.flash(c, "#000000", pp["fade"])
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
        print(f"{path}  {1000 * (time.time() - t0):.0f} ms", flush=True)


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


def video(t_from=0.0, t_to=None, workers=4, out=None, audio_path=None):
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
    out = out or os.path.join(ROOT, "out", "chase_200_sub_special.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    song = audio_path or os.path.join(BUILD, "song.mp3")
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
