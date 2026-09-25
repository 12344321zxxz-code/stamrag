"""Audio analysis: beat grid + per-frame envelopes saved to build/features.npz.

    python -m mv.audio path/to/song.mp3
"""
import os
import subprocess
import sys

import librosa
import numpy as np

from .core import BUILD


def decode(path, sr=22050):
    wav = os.path.join(BUILD, "song_mono.wav")
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                        "-ac", "1", "-ar", str(sr), wav], check=True)
    y, _ = librosa.load(wav, sr=sr)
    return y, sr


def beat_grid(y, sr, hop=256):
    """Tracked beats, regularised onto a locally-corrected constant-tempo grid."""
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop,
                                       units="time", start_bpm=103)
    beats = np.asarray(beats)
    idx = np.arange(len(beats))
    period, offset = np.polyfit(idx, beats, 1)
    # re-fit on the well-behaved part (skip intro drift)
    k = np.round((beats - offset) / period).astype(int)
    res = beats - (offset + period * k)
    good = np.abs(res) < 0.12
    period, offset = np.polyfit(k[good], beats[good], 1)
    res = beats - (offset + period * k)
    good = np.abs(res) < 0.12
    kk, rr = k[good], res[good]
    # phase so that grid index 0 is a bar downbeat (found by section alignment: see README)
    n = int((len(y) / sr - offset) / period) + 3
    grid = np.zeros(n)
    for i in range(n):
        m = np.abs(kk - i) <= 8
        corr = np.median(rr[m]) if m.sum() >= 3 else np.median(rr[np.argsort(np.abs(kk - i))[:6]])
        grid[i] = offset + period * i + corr
    return grid


def envelopes(y, sr, fps=100):
    hop = int(sr / fps)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    ft = librosa.times_like(S[0], sr=sr, hop_length=hop)

    def band(lo, hi):
        return S[(freqs >= lo) & (freqs < hi)].sum(0)

    def norm(x, q=97):
        return np.clip(x / (np.percentile(x, q) + 1e-9), 0, 1.5)

    def flux(x):
        d = np.maximum(0, np.diff(np.log1p(x * 10), prepend=np.log1p(x[0] * 10)))
        return d

    def follow(x, att=0.6, rel=0.12):
        out = np.zeros_like(x)
        v = 0.0
        for i, s in enumerate(x):
            v = v + (s - v) * (att if s > v else rel)
            out[i] = v
        return out

    low, mid, high = band(25, 160), band(160, 2500), band(2500, 11000)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0][: len(ft)]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)[: len(ft)]
    kick = norm(follow(norm(flux(band(30, 120)), 99), 0.9, 0.18), 99)
    snare = norm(follow(norm(flux(band(1500, 5000)), 99), 0.9, 0.2), 99)
    hat = norm(follow(norm(flux(band(7000, 11000)), 99), 0.9, 0.25), 99)

    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=32, fmax=11000)
    mel = librosa.power_to_db(mel, ref=np.max)
    spec = np.clip((mel + 70) / 70, 0, 1).T
    # smooth spectrum in time (attack fast / release slow)
    sm = np.zeros_like(spec)
    v = np.zeros(spec.shape[1])
    for i in range(len(spec)):
        a = np.where(spec[i] > v, 0.7, 0.15)
        v = v + (spec[i] - v) * a
        sm[i] = v

    return dict(ft=ft, low=norm(follow(low)), mid=norm(follow(mid)), high=norm(follow(high)),
                rms=norm(follow(rms, 0.5, 0.08)), onset=norm(onset, 99), kick=kick, snare=snare,
                hat=hat, spec=sm.astype(np.float32))


def main(path):
    os.makedirs(BUILD, exist_ok=True)
    y, sr = decode(path)
    grid = beat_grid(y, sr)
    env = envelopes(y, sr)
    np.savez(os.path.join(BUILD, "features.npz"), grid=grid, duration=len(y) / sr, **env)
    print(f"duration {len(y)/sr:.2f}s  beats {len(grid)}  bpm {60/np.median(np.diff(grid)):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
