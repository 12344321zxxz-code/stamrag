"""The cast's voices: Kokoro-82M text-to-speech (kokoro-onnx), shifted into cartoon registers.

A voice is (kokoro voice, semitones, rate, language). Pitch is shifted by resampling, which moves the
formants too: up for the small critters, down for the whale. Kokoro renders the line slower by the
same ratio first, so the speaking rate comes out as `rate`. Clips are cached in build_paper/voice/.
"""
import hashlib
import os
import subprocess
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

from . import script
from .core import BUILD, SR, build_path

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
TTS_SR = 24000
_kokoro = None


def _model(name):
    path = build_path("models", name)
    if not os.path.exists(path):
        print("downloading", name, flush=True)
        subprocess.run(["curl", "-sSL", "--fail", "-o", path + ".part", MODEL_URL + name], check=True)
        os.replace(path + ".part", path)
    return path


def kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(_model(MODEL_FILES[0]), _model(MODEL_FILES[1]))
    return _kokoro


def _resample(x, sr_from, sr_to):
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(x, int(sr_to) // g, int(sr_from) // g).astype(np.float32)


def _trim(x, sr, thresh_db=-42.0, pre=0.03, post=0.09):
    env = np.sqrt(np.convolve(x * x, np.ones(256) / 256, mode="same"))
    on = np.nonzero(env > env.max() * 10 ** (thresh_db / 20))[0]
    if len(on) == 0:
        return x
    a = max(0, on[0] - int(pre * sr))
    b = min(len(x), on[-1] + int(post * sr))
    y = x[a:b].copy()
    n = int(0.006 * sr)
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def _level(x, target_db=-19.0):
    """Scale so the loud (voiced) part of the clip sits at target_db RMS."""
    fr = 1024
    n = len(x) // fr
    if n == 0:
        return x
    rms = np.sqrt((x[:n * fr].reshape(n, fr) ** 2).mean(1))
    loud = rms[rms > rms.max() * 0.25]
    g = 10 ** (target_db / 20) / (np.sqrt((loud ** 2).mean()) + 1e-9)
    y = x * g
    pk = np.abs(y).max()
    return y * (0.89 / pk) if pk > 0.89 else y


def render(member, text):
    """-> float32 mono clip at SR for one cast member saying text."""
    v, semis, rate, lang = script.VOICES[member]
    key = hashlib.sha1(repr((v, semis, rate, lang, text, 3)).encode()).hexdigest()[:16]
    path = os.path.join(BUILD, "voice", f"{member}_{key}.wav")
    if os.path.exists(path):
        return sf.read(path, dtype="float32")[0]
    r = 2 ** (semis / 12)
    x, sr = kokoro().create(text, voice=v, speed=float(np.clip(rate / r, 0.5, 2.0)), lang=lang)
    x = np.asarray(x, dtype=np.float32)
    if semis:
        up, down = 1000, int(round(1000 * r))
        g = gcd(up, down)
        x = resample_poly(x, up // g, down // g).astype(np.float32)
    x = _trim(x, sr)
    x = sosfilt(butter(2, 90 if semis >= 0 else 60, "highpass", fs=sr, output="sos"), x).astype(np.float32)
    x = _level(_resample(x, sr, SR))
    sf.write(build_path("voice", os.path.basename(path)), x, SR)
    return x


def check(clips):
    """Round-trip every clip through Whisper (sherpa-onnx) and print what it hears; needs the optional
    model in build_paper/models/sherpa-onnx-whisper-base.en."""
    import sherpa_onnx
    d = os.path.join(BUILD, "models", "sherpa-onnx-whisper-base.en")
    if not os.path.isdir(d):
        print("skipping ASR check (no", d, ")")
        return
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=os.path.join(d, "base.en-encoder.int8.onnx"), decoder=os.path.join(d, "base.en-decoder.int8.onnx"),
        tokens=os.path.join(d, "base.en-tokens.txt"), language="en", task="transcribe", num_threads=4)
    for member, text, x in clips:
        s = rec.create_stream()
        s.accept_waveform(16000, _resample(x, SR, 16000))
        rec.decode_stream(s)
        print(f"  {member:9s} {text!r:60s} -> {s.result.text.strip()!r}")
