import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from intro import play_intro_scene



class FullVideo(Scene):
    def construct(self):
        # --- 1. INTRODUCTION ---
        play_intro_scene(self)

        # --- 2. LA SUITE (exemple pour plus tard) ---
        # from part2_unet import play_unet_scene
        # play_unet_scene(self)

        # --- 3. CONCLUSION ---
        # from outro import play_outro_scene
        # play_outro_scene(self)
