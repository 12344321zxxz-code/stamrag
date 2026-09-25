"""Audio analysis for the CHASE clip -> build_chase/features.npz

    python -m chase.audio build_chase/song.mp3

Extracts: a constant-tempo beat grid (112.1 BPM, beat 0 = band entry), drum / bass / level
envelopes, the talking-intro speech envelope + phrase boundaries, the whammy-dive pitch curve,
and every guitar note transcribed by Spotify's basic-pitch (plus a 'lead' top-voice subset).
"""
import os
import subprocess
import sys

import librosa
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build_chase")
SR = 22050

# Found by comb-filtering the percussive onset envelope (see README): period and first downbeat.
PERIOD = 0.53525
BEAT0 = 24.216


def decode(path):
    wav = os.path.join(BUILD, "song_mono.wav")
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-ac", "1",
                        "-ar", str(SR), wav], check=True)
    y, _ = librosa.load(wav, sr=SR)
    return y


def refine_grid(y, n_before=48):
    """Constant grid, nudged per 16-beat window toward the strongest percussive alignment."""
    hop = 256
    _, yp = librosa.effects.hpss(y, margin=2.0)
    oenv = librosa.onset.onset_strength(y=yp, sr=SR, hop_length=hop)
    fps = SR / hop
    n = int((len(y) / SR - BEAT0) / PERIOD) + 2
    grid = BEAT0 + PERIOD * np.arange(n, dtype=float)
    corr = np.zeros(n)
    for w in range(0, n, 16):
        ks = np.arange(w, min(n, w + 16))
        best = (-1, 0.0)
        for d in np.arange(-0.03, 0.0301, 0.002):
            idx = np.clip(np.round((grid[ks] + d) * fps).astype(int), 0, len(oenv) - 1)
            v = oenv[idx].mean()
            if v > best[0]:
                best = (v, d)
        corr[ks] = best[1]
    # smooth the per-window corrections so the grid never jumps
    from scipy.ndimage import gaussian_filter1d
    grid = grid + gaussian_filter1d(corr, 6)
    before = BEAT0 - PERIOD * np.arange(n_before, 0, -1)
    return np.concatenate([before, grid]), n_before


def envelopes(y, fps=100):
    hop = int(SR / fps)
    yh, yp = librosa.effects.hpss(y, margin=2.0)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    Sp = np.abs(librosa.stft(yp, n_fft=2048, hop_length=hop))
    Sh = np.abs(librosa.stft(yh, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    ft = librosa.times_like(S[0], sr=SR, hop_length=hop)

    def band(X, lo, hi):
        return X[(freqs >= lo) & (freqs < hi)].sum(0)

    def norm(x, q=97):
        return np.clip(x / (np.percentile(x, q) + 1e-9), 0, 1.5)

    def flux(x):
        lx = np.log1p(x * 10)
        return np.maximum(0, np.diff(lx, prepend=lx[0]))

    def follow(x, att=0.6, rel=0.12):
        out = np.zeros_like(x)
        v = 0.0
        for i, s in enumerate(x):
            v = v + (s - v) * (att if s > v else rel)
            out[i] = v
        return out

    kick = norm(follow(norm(flux(band(Sp, 30, 120)), 99), 0.9, 0.2), 99)
    snare = norm(follow(norm(flux(band(Sp, 1500, 5000)), 99), 0.9, 0.22), 99)
    hat = norm(follow(norm(flux(band(Sp, 7000, 11000)), 99), 0.9, 0.3), 99)
    bass = norm(follow(band(Sh, 30, 180), 0.5, 0.1))
    gtr = norm(follow(band(Sh, 180, 4000), 0.5, 0.1))
    rms = norm(follow(librosa.feature.rms(y=y, hop_length=hop)[0][: len(ft)], 0.5, 0.08))
    onset = norm(librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)[: len(ft)], 99)
    # speech envelope (only meaningful before the band enters)
    sp = band(S, 250, 3500)
    from scipy.ndimage import uniform_filter1d
    sp = uniform_filter1d(sp, 5)
    m = ft < BEAT0
    base, peak = np.percentile(sp[m], 20), np.percentile(sp[m], 95)
    speech = np.clip((sp - base) / (peak - base + 1e-9), 0, 1.3) * m
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=32, fmax=11000), ref=np.max)
    spec = np.clip((mel + 70) / 70, 0, 1).T.astype(np.float32)
    return dict(ft=ft, kick=kick, snare=snare, hat=hat, bass=bass, gtr=gtr, rms=rms, onset=onset,
                speech=speech, spec=spec)


def dive_curve(y):
    """Pitch (MIDI) of the whammy dive + slide that lead into the band entry."""
    t0, t1 = 16.5, 24.3
    seg = y[int(t0 * SR):int(t1 * SR)]
    f0, _, _ = librosa.pyin(seg, fmin=55, fmax=1200, sr=SR, hop_length=256)
    tt = t0 + np.arange(len(f0)) * 256 / SR
    midi = np.where(np.isnan(f0), 0.0, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)))
    return tt, midi


def notes(path):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    _, _, ev = predict(path, ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                       minimum_note_length=80, minimum_frequency=60, maximum_frequency=1500)
    ev = sorted((float(s), float(e), int(p), float(a)) for (s, e, p, a, _) in ev)
    arr = np.array(ev, dtype=float)
    # lead = highest note among those starting within 40 ms of each other, pitch >= 50
    lead = []
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[j + 1, 0] - arr[i, 0] < 0.04:
            j += 1
        grp = arr[i:j + 1]
        top = grp[np.argmax(grp[:, 2])]
        if top[2] >= 50:
            lead.append(top)
        i = j + 1
    return arr, np.array(lead)


# Talking-intro phrase boundaries (deepest dips of the speech envelope; see README)
SPEECH_PHRASES = [(2.70, 4.52), (4.52, 6.26), (6.26, 7.71), (7.71, 9.72), (9.72, 11.38), (11.38, 14.23),
                  (14.23, 16.21)]


def main(path):
    os.makedirs(BUILD, exist_ok=True)
    y = decode(path)
    grid, n_before = refine_grid(y)
    env = envelopes(y)
    dt, dm = dive_curve(y)
    allnotes, lead = notes(path)
    np.savez(os.path.join(BUILD, "features.npz"), grid=grid, n_before=n_before, duration=len(y) / SR,
             dive_t=dt, dive_m=dm, notes=allnotes, lead=lead, phrases=np.array(SPEECH_PHRASES), **env)
    print(f"duration {len(y)/SR:.2f}s  beats {len(grid)}  notes {len(allnotes)}  lead {len(lead)}")


if __name__ == "__main__":
    main(sys.argv[1])
