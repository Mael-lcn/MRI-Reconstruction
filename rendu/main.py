import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manimlib import *

from intro import play_intro_problem_scene
from mri_data import play_ifft_rss_scene
from diffusion import play_intro_scene, play_forward_math_scene, play_reverse_math_scene, play_loss_scene
from archi import play_diffcmr_architecture_scene
from flow import play_flow_matching_euler_scene
from res import play_training_evolution_scene, play_quantitative_comparison_scene



class DiffCMR(Scene):
    def construct(self):
        self.camera.background_rgba = [0.08, 0.08, 0.08, 1.0]

        # --- Intro ---
        play_intro_problem_scene(self)
        play_ifft_rss_scene(self)

        # --- Diffusion ---
        play_intro_scene(self)
        play_forward_math_scene(self)
        play_reverse_math_scene(self)
        play_loss_scene(self)

        # --- model ---
        play_diffcmr_architecture_scene(self)

        # --- Flow ---
        play_flow_matching_euler_scene(self)

        # --- resultats ----
        play_training_evolution_scene(self)
        play_quantitative_comparison_scene(self)
