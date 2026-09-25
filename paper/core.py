"""Constants and small helpers shared by the audio build and the renderer."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build_paper")
FONT_DIR = os.path.join(ROOT, "assets", "fonts")
W, H = 1920, 1080
FPS = 30
SR = 48000


def build_path(*parts):
    p = os.path.join(BUILD, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
