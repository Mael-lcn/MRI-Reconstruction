from manimlib import *



def play_training_evolution_scene(scene):
    # --- 1. TITRE ---
    title = Text("Model Training: Validation Set Evolution", color=TEAL).to_edge(UP)
    scene.play(Write(title))
    scene.wait(0.5)

    # Ligne d'alignement globale pour toutes les images (pour être sûr que tout est droit)
    y_level = DOWN * 0.5 

    # =========================================================
    # MISE EN PLACE DES RÉFÉRENCES (GAUCHE / DROITE)
    # =========================================================
    # IMAGE INPUT
    img_in = ImageMobject("images/train/input_b0_i1.png").set_height(3.5).move_to(LEFT * 4.5 + y_level)
    lbl_in = Tex(r"\text{Input (Zero-Filled)}", font_size=32, color=RED).next_to(img_in, DOWN, buff=0.4)

    # IMAGE GROUND TRUTH
    img_gt = ImageMobject("images/train/gt_b0_i1.png").set_height(3.5).move_to(RIGHT * 4.5 + y_level)
    lbl_gt = Tex(r"\text{Ground Truth}", font_size=32, color=GREEN).next_to(img_gt, DOWN, buff=0.4)

    # Apparition des références
    scene.play(FadeIn(img_in, RIGHT), Write(lbl_in))
    scene.play(FadeIn(img_gt, LEFT), Write(lbl_gt))
    scene.wait(1)

    # =========================================================
    # L'ÉVOLUTION AU CENTRE (10k -> 20k -> 30k -> 150k)
    # =========================================================
    center_title = Text("DiffCMR Prediction", font_size=32, color=YELLOW).next_to(title, DOWN, buff=0.4)
    scene.play(Write(center_title))

    # Liste de tes checkpoints avec le texte associé
    train_steps = [
        ("images/train/recon_b0_i1_10k.png", "Iteration: 10k"),
        ("images/train/recon_b0_i1 copy_20k.png", "Iteration: 20k"),
        ("images/train/recon_b0_i1_30K.png", "Iteration: 30k"),
        ("images/train/recon_b0_i1_150K.png", "Iteration: 150k (Converged)"),
    ]

    center_pos = y_level # On utilise exactement la même hauteur que les côtés
    current_img = None
    current_lbl = None

    for i, (path, text) in enumerate(train_steps):
        # Chargement de la nouvelle image
        new_img = ImageMobject(path).set_height(3.5).move_to(center_pos)

        # Le label change de couleur à la fin pour marquer la convergence
        lbl_color = YELLOW if i < len(train_steps) - 1 else TEAL
        new_lbl = Tex(r"\text{" + text + "}", font_size=36, color=lbl_color).next_to(new_img, DOWN, buff=0.4)

        if current_img is None:
            # Première image (10k) : on la fait apparaître simplement
            scene.play(FadeIn(new_img, UP), Write(new_lbl))
        else:
            # Effet de "Crossfade" (Fondu enchaîné)
            scene.play(
                FadeOut(current_img),
                FadeOut(current_lbl),
                FadeIn(new_img),
                FadeIn(new_lbl),
                run_time=1.2 
            )

        current_img = new_img
        current_lbl = new_lbl
        scene.wait(1.5)

    # =========================================================
    # HIGHLIGHT FINAL DE LA CONVERGENCE
    # =========================================================
    # On encadre la prédiction finale pour montrer qu'elle est prête
    box_final = SurroundingRectangle(current_img, color=TEAL, buff=0.0)


    match_arrow = Arrow(current_img.get_right(), img_gt.get_left(), buff=0.15, color=TEAL)
    match_text = Text("Matches!", font_size=24, color=TEAL).next_to(match_arrow, UP, buff=0.1)

    scene.play(ShowCreation(box_final), GrowArrow(match_arrow))
    scene.play(Write(match_text))
    
    scene.wait(4)

    scene.play(FadeOut(Group(*scene.mobjects)))
    scene.clear()




def play_quantitative_comparison_scene(scene):
    # --- 1. TITRE DE TRANSITION ---
    title = Text("From Diffusion to Flow Matching", color=TEAL).to_edge(UP)
    scene.play(Write(title))
    scene.wait(0.5)

    # =========================================================
    # PARTIE 1 : COMPARAISON VISUELLE
    # =========================================================
    visual_text = Text("Visually equivalent...", font_size=32, color=GREY_A).next_to(title, DOWN, buff=0.5)
    scene.play(Write(visual_text))

    img_diff = ImageMobject("images/eval/recon_b0_i1_diff.png").set_height(3.5)
    img_flow = ImageMobject("images/eval/recon_b0_i1_flow.png").set_height(3.5)
    img_gt = ImageMobject("images/eval/gt_b0_i1.png").set_height(3.5)

    images_group = Group(img_diff, img_flow, img_gt).arrange(RIGHT, buff=0.8).shift(DOWN * 0.5)

    lbl_diff = Tex(r"\text{DiffCMR}", font_size=32, color=BLUE).next_to(img_diff, DOWN)
    lbl_flow = Tex(r"\textbf{Flow Matching (Ours)}", font_size=36, color=YELLOW).next_to(img_flow, DOWN)
    lbl_gt = Tex(r"\text{Ground Truth}", font_size=32, color=GREEN).next_to(img_gt, DOWN)

    scene.play(FadeIn(img_diff, UP), Write(lbl_diff))
    scene.play(FadeIn(img_flow, UP), Write(lbl_flow))
    scene.play(FadeIn(img_gt, UP), Write(lbl_gt))
    scene.wait(1)

    # Transition vers les chiffres
    scene.play(FadeOut(images_group), FadeOut(lbl_diff), FadeOut(lbl_flow), FadeOut(lbl_gt))
    
    trans_text = Text("...but what about computational efficiency?", font_size=32, color=YELLOW)
    trans_text.move_to(visual_text.get_center())
    scene.play(ReplacementTransform(visual_text, trans_text))
    scene.wait(1.5)

    scene.play(FadeOut(trans_text))

    # =========================================================
    # PARTIE 2 : LE NOMBRE DE STEPS (T)
    # =========================================================
    # Explication de T
    def_T = Tex(r"\textbf{Parameter } T \textbf{: Number of Integration Steps}", font_size=36, color=BLUE).next_to(title, DOWN, buff=0.3)
    scene.play(Write(def_T))

    # Affichage de ton tableau T sous forme d'image
    img_T = ImageMobject("images/eval/test_T.png").set_height(4.5).next_to(def_T, DOWN, buff=0.4)
    frame_T = SurroundingRectangle(img_T, color=WHITE, stroke_width=2, buff=0) # Joli cadre
    
    scene.play(FadeIn(img_T, UP), ShowCreation(frame_T))
    scene.wait(1)

    # Highlight (texte flottant pour attirer l'attention sur la conclusion)
    speed_text = Text("Flow Matching requires only 40 steps vs 1000!\n(~250x Faster)", font_size=32, color=GREEN, alignment="CENTER")
    speed_text.next_to(img_T, DOWN, buff=0.3)

    scene.play(Write(speed_text))
    scene.wait(2)

    # Nettoyage
    scene.play(FadeOut(Group(def_T, img_T, frame_T, speed_text)))

    # =========================================================
    # PARTIE 3 : LE NOMBRE DE ROUNDS (R)
    # =========================================================
    # Explication de R
    def_R = Tex(r"\textbf{Parameter } R \textbf{: Number of Rounds (Ensemble Averaging)}", font_size=36, color=PURPLE_B).next_to(title, DOWN, buff=0.3)
    scene.play(Write(def_R))

    # Affichage de ton tableau R (positionné par rapport à def_R maintenant)
    img_R = ImageMobject("images/eval/test_R.png").set_height(4.5).next_to(def_R, DOWN, buff=0.4)
    frame_R = SurroundingRectangle(img_R, color=WHITE, stroke_width=2, buff=0)

    scene.play(FadeIn(img_R, UP), ShowCreation(frame_R))
    
    scene.wait(3) 

    # Nettoyage sans les anciennes variables
    scene.play(FadeOut(Group(def_R, img_R, frame_R)))

    # =========================================================
    # PARTIE 4 : CONCLUSION GLOBALE
    # =========================================================
    def_G = Tex(r"\textbf{Global Comparison}", font_size=40, color=GOLD).next_to(title, DOWN, buff=0.3)
    scene.play(Write(def_G))

    # Affichage global
    img_G = ImageMobject("images/eval/global.png").set_height(5.0).next_to(def_G, DOWN, buff=0.3)
    frame_G = SurroundingRectangle(img_G, color=GOLD, stroke_width=3, buff=0)
    
    scene.play(FadeIn(img_G, scale=1.1), ShowCreation(frame_G))
    scene.wait(4)

    # Rendu final
    scene.play(FadeOut(Group(*scene.mobjects)))
    scene.clear()
