import os
import cv2
import tempfile
import numpy as np
from manimlib import *



# ==========================================
# LECTEUR VIDÉO CUSTOM POUR MANIMGL
# ==========================================
class VideoPlayer(Group):
    """
    Lit une vidéo mp4 frame-par-frame avec OpenCV.
    Synchronise la lecture sur le temps virtuel de Manim pour une vitesse constante.
    """
    def __init__(self, filename, height=4.5, fps=15, **kwargs):
        super().__init__(**kwargs)
        self.cap = cv2.VideoCapture(filename)
        self.vid_height = height
        self.frame_count = 0

        # --- GESTION DU TEMPS POUR UNE LECTURE RÉGULIÈRE ---
        self.fps = fps                           # Vitesse désirée (ex: 15 images par seconde)
        self.frame_duration = 1.0 / self.fps     # Temps que doit durer une frame (ex: 0.066s)
        self.time_passed = 0.0                   # Accumulateur de temps Manim

        # Création d'un dossier temporaire dédié
        self.temp_dir = tempfile.mkdtemp()
        self.current_temp = os.path.join(self.temp_dir, f"frame_{self.frame_count}.jpg")
        
        # Test de lecture
        ret, frame = self.cap.read()
        if not ret:
            print(f"ERREUR CRITIQUE: Impossible de lire la vidéo {filename} !")
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # 1. Sauvegarde de la première frame
        cv2.imwrite(self.current_temp, frame)

        # 2. Chargement initial
        self.img_mob = ImageMobject(self.current_temp)
        self.img_mob.set_height(self.vid_height)
        self.add(self.img_mob)

        # 3. Activation de l'updater
        self.add_updater(self.update_frame)

    def update_frame(self, mob, dt):
        # On ajoute le temps écoulé depuis le dernier calcul de Manim
        mob.time_passed += dt

        # Si le temps accumulé dépasse la durée prévue pour une image vidéo
        if mob.time_passed >= mob.frame_duration:
            # On calcule combien de frames on doit lire pour "rattraper" le temps
            frames_to_read = int(mob.time_passed / mob.frame_duration)

            # On conserve le "reste" du temps pour le prochain cycle (modulo)
            mob.time_passed = mob.time_passed % mob.frame_duration

            ret = False
            frame = None

            # On lit la vidéo du nombre de frames calculé
            for _ in range(frames_to_read):
                ret, frame = mob.cap.read()
                if not ret:
                    # Fin de la vidéo -> on boucle !
                    mob.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = mob.cap.read()
                    break # On arrête la boucle for pour éviter un bug

            if ret and frame is not None:
                mob.frame_count += 1
                new_temp = os.path.join(mob.temp_dir, f"frame_{mob.frame_count}.jpg")

                # Écriture de la nouvelle frame
                cv2.imwrite(new_temp, frame)

                # Chargement forcé par Manim
                new_img = ImageMobject(new_temp)
                new_img.set_height(mob.vid_height)
                new_img.move_to(mob.img_mob.get_center())
                
                mob.remove(mob.img_mob)
                mob.img_mob = new_img
                mob.add(mob.img_mob)

                # Nettoyage de l'ancienne frame
                try:
                    old_temp = os.path.join(mob.temp_dir, f"frame_{mob.frame_count - 1}.jpg")
                    if os.path.exists(old_temp):
                        os.remove(old_temp)
                except Exception:
                    pass



def play_intro_problem_scene(scene):
    # --- 1. LE CONTEXTE CLINIQUE ---
    title = Text("The Clinical Challenge: Cardiac MRI", color=GOLD).to_edge(UP, buff=0.3)
    scene.play(Write(title))

    clinical_points = VGroup(
        Text("1. Gold Standard for cardiac function and morphology", font_size=28, color=WHITE),
        Text("2. Very long acquisition time (45 - 60 minutes)", font_size=28, color=RED_B),
        Text("3. Uncomfortable for patients (multiple breath-holds)", font_size=28, color=RED_B)
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(UP * 1.0)

    for point in clinical_points:
        scene.play(Write(point), run_time=0.8)
    scene.wait(1.5)

    acc_solution = Text("Solution: K-Space Undersampling (Acceleration Factor 4x)", font_size=32, color=BLUE_B)
    acc_solution.next_to(clinical_points, DOWN, buff=0.8)
    
    acc_problem = Text("Consequence: Severe aliasing artifacts in the spatial domain", font_size=28, color=GREY)
    acc_problem.next_to(acc_solution, DOWN, buff=0.3)

    scene.play(Write(acc_solution))
    scene.play(FadeIn(acc_problem, DOWN))
    scene.wait(2.5)

    # --- TRANSITION ---
    scene.play(
        FadeOut(clinical_points),
        FadeOut(acc_solution),
        FadeOut(acc_problem),
        title.animate.become(Text("Impact on Downstream Tasks (MedSAM 2)", color=GOLD).to_edge(UP, buff=0.3))
    )

    # --- 2. LECTURE DES VIDÉOS BRUTES ---
    path_full = "../../preuve/FullSample_Video.mp4"
    path_acc = "../../preuve/AccFactor04_Video.mp4"

    vid_full = VideoPlayer(path_full, height=4.5).move_to(LEFT * 3.5 + DOWN * 0.5)
    vid_acc = VideoPlayer(path_acc, height=4.5).move_to(RIGHT * 3.5 + DOWN * 0.5)

    gt_title = Text("Fully Sampled (Ground Truth)", font_size=28, color=GREEN_B).next_to(vid_full, UP, buff=0.3)
    acc_title = Text("Accelerated (Acc 04 - Aliased)", font_size=28, color=RED_B).next_to(vid_acc, UP, buff=0.3)

    scene.play(FadeIn(vid_full), FadeIn(vid_acc), Write(gt_title), Write(acc_title))

    explanation = Text("Without reconstruction, images are heavily corrupted.", font_size=26, color=GREY).to_edge(DOWN, buff=0.5)
    scene.play(Write(explanation))

    scene.wait(2)

    # --- 3. LECTURE DES VIDÉOS SEGMENTÉES ---
    sam_text = Text("Applying MedSAM 2...", font_size=28, color=YELLOW).move_to(explanation)
    scene.play(ReplacementTransform(explanation, sam_text))
    scene.wait(1.5)

    path_clean = "../../preuve/clean_data.mp4"
    path_crap = "../../preuve/crap_data.mp4"

    vid_clean = VideoPlayer(path_clean, height=4.5).move_to(vid_full.get_center())
    vid_crap = VideoPlayer(path_crap, height=4.5).move_to(vid_acc.get_center())

    gt_success = Text("Successful Segmentation", font_size=24, color=GREEN_B).next_to(vid_clean, DOWN, buff=0.3)
    acc_fail = Text("Segmentation Failed (Complete Crash)", font_size=24, color=RED_B).next_to(vid_crap, DOWN, buff=0.3)

    # Remplacement en fondu
    scene.play(
        FadeOut(vid_full), FadeOut(vid_acc),
        FadeIn(vid_clean), FadeIn(vid_crap)
    )
    scene.play(Write(gt_success), Write(acc_fail))

    # On souligne l'échec pour la conclusion
    scene.play(Indicate(acc_fail, color=RED_B, scale_factor=1.1), run_time=1.5)
    
    scene.wait(2)

    # --- CONCLUSION ---
    conclusion = Text("Conclusion: Robust image reconstruction is mandatory.", font_size=32, color=GOLD).move_to(sam_text)
    
    scene.play(ReplacementTransform(sam_text, conclusion))
    scene.wait(3.5)

    # Nettoyage final
    scene.play(FadeOut(Group(*scene.mobjects)))
