from manimlib import *



def play_training_evolution_scene(scene):
    # =========================================================
    # TITRE ET RÉFÉRENCES (Stables sur les côtés)
    # =========================================================
    title = Text("Model Training: Validation Set Evolution", color=TEAL).to_edge(UP)
    scene.play(Write(title))

    y_level = DOWN * 0.5 

    # IMAGE INPUT (Gauche)
    img_in = ImageMobject("images/train/input_b0_i1.png").set_height(3.5).move_to(LEFT * 4.5 + y_level)
    lbl_in = Tex(r"\text{Input (Zero-Filled)}", font_size=32, color=RED).next_to(img_in, DOWN, buff=0.4)

    # IMAGE GROUND TRUTH (Droite)
    img_gt = ImageMobject("images/train/gt_b0_i1.png").set_height(3.5).move_to(RIGHT * 4.5 + y_level)
    lbl_gt = Tex(r"\text{Ground Truth}", font_size=32, color=GREEN).next_to(img_gt, DOWN, buff=0.4)

    scene.play(FadeIn(img_in, RIGHT), Write(lbl_in), FadeIn(img_gt, LEFT), Write(lbl_gt))
    scene.wait(0.5)

    # =========================================================
    # L'ÉVOLUTION AU CENTRE AVEC JAUGE DE PROGRESSION
    # =========================================================
    center_title = Text("DiffCMR Prediction", font_size=32, color=YELLOW).next_to(title, DOWN, buff=0.4)
    scene.play(Write(center_title))

    train_steps = [
        ("images/train/recon_b0_i1_10k.png", "10k", 0.1),
        ("images/train/recon_b0_i1_20k.png", "20k", 0.2),
        ("images/train/recon_b0_i1_30K.png", "30k", 0.3),
        ("images/train/recon_b0_i1_150K.png", "150k", 1.0),
    ]

    center_pos = y_level
    current_img = None
    current_lbl = None

    # Barre de progression visuelle pro
    progress_bar_bg = Rectangle(width=4, height=0.1, color=GREY_E, fill_opacity=1).move_to(y_level + DOWN * 2.5)
    scene.play(ShowCreation(progress_bar_bg))

    for i, (path, text, progress) in enumerate(train_steps):
        new_img = ImageMobject(path).set_height(3.5).move_to(center_pos)
        
        is_final = (i == len(train_steps) - 1)
        lbl_color = TEAL if is_final else YELLOW
        lbl_text = f"Iteration: {text}" + (" (Converged)" if is_final else "")
        new_lbl = Text(lbl_text, font_size=24, color=lbl_color).next_to(new_img, DOWN, buff=0.3)

        # Remplissage de la barre
        progress_fill = Rectangle(width=4 * progress, height=0.1, color=lbl_color, fill_opacity=1, stroke_width=0)
        progress_fill.align_to(progress_bar_bg, LEFT).match_y(progress_bar_bg)

        if current_img is None:
            scene.play(FadeIn(new_img, UP), Write(new_lbl), FadeIn(progress_fill))
        else:
            scene.play(
                FadeOut(current_img), FadeOut(current_lbl),
                FadeIn(new_img), FadeIn(new_lbl),
                Transform(current_progress, progress_fill),
                run_time=1.0 
            )

        current_img = new_img
        current_lbl = new_lbl
        current_progress = progress_fill
        scene.wait(1)

    scene.wait(1)
    scene.play(FadeOut(Group(*scene.mobjects)))




def play_training_evolution_scene(scene):
    title = Text("Model Training: Validation Set Evolution", color=TEAL).to_edge(UP)
    scene.play(Write(title))

    y_level = DOWN * 0.5 

    # RÉFÉRENCES STABLES
    img_in = ImageMobject("images/train/input_b0_i1.png").set_height(3).move_to(LEFT * 5 + y_level)
    lbl_in = Text("Input (Zero-Filled)", font_size=20, color=RED).next_to(img_in, DOWN)

    img_gt = ImageMobject("images/train/gt_b0_i1.png").set_height(3).move_to(RIGHT * 5 + y_level)
    lbl_gt = Text("Ground Truth", font_size=20, color=GREEN).next_to(img_gt, DOWN)

    scene.play(FadeIn(img_in, RIGHT), Write(lbl_in), FadeIn(img_gt, LEFT), Write(lbl_gt))

    # ÉVOLUTION AU CENTRE
    center_title = Text("DiffCMR Prediction", font_size=28, color=YELLOW).next_to(title, DOWN, buff=0.4)
    scene.play(Write(center_title))

    train_steps = [
        ("images/train/recon_b0_i1_10k.png", "10k", 0.1),
        ("images/train/recon_b0_i1 copy_20k.png", "20k", 0.2),
        ("images/train/recon_b0_i1_30K.png", "30k", 0.3),
        ("images/train/recon_b0_i1_150K.png", "150k", 1.0),
    ]

    center_pos = y_level
    current_img = None
    current_lbl = None
    current_progress = None
    
    # Barre de progression
    progress_bar_bg = Rectangle(width=4, height=0.1, color=GREY_E, fill_opacity=1).move_to(y_level + DOWN * 2.5)
    scene.play(ShowCreation(progress_bar_bg))

    for i, (path, text, progress) in enumerate(train_steps):
        new_img = ImageMobject(path).set_height(3).move_to(center_pos)
        
        is_final = (i == len(train_steps) - 1)
        lbl_color = TEAL if is_final else YELLOW
        lbl_text_s = f"Iteration: {text}" + (" (Converged)" if is_final else "")
        new_lbl = Text(lbl_text_s, font_size=20, color=lbl_color).next_to(new_img, DOWN, buff=0.3)
        
        # Remplissage de la barre
        progress_fill = Rectangle(width=4 * progress, height=0.1, color=lbl_color, fill_opacity=1, stroke_width=0)
        progress_fill.align_to(progress_bar_bg, LEFT).match_y(progress_bar_bg)

        if current_img is None:
            scene.play(FadeIn(new_img, UP), Write(new_lbl), FadeIn(progress_fill))
        else:
            scene.play(
                FadeOut(current_img), FadeOut(current_lbl),
                FadeIn(new_img), FadeIn(new_lbl),
                Transform(current_progress, progress_fill),
                run_time=1.0 
            )

        current_img = new_img
        current_lbl = new_lbl
        current_progress = progress_fill
        scene.wait(1)

    box_final = SurroundingRectangle(current_img, color=TEAL, buff=0.0)
    scene.play(ShowCreation(box_final), Flash(current_lbl, color=TEAL))
    
    scene.wait(2)
    scene.play(FadeOut(Group(*scene.mobjects)))





def play_training_evolution_scene(scene):
    title = Text("Model Training: Validation Set Evolution", color=TEAL).to_edge(UP)
    scene.play(Write(title))

    y_level = DOWN * 0.5 

    # RÉFÉRENCES STABLES
    img_in = ImageMobject("images/train/input_b0_i1.png").set_height(3).move_to(LEFT * 5 + y_level)
    lbl_in = Text("Input (Zero-Filled)", font_size=20, color=RED).next_to(img_in, DOWN)

    img_gt = ImageMobject("images/train/gt_b0_i1.png").set_height(3).move_to(RIGHT * 5 + y_level)
    lbl_gt = Text("Ground Truth", font_size=20, color=GREEN).next_to(img_gt, DOWN)

    scene.play(FadeIn(img_in, RIGHT), Write(lbl_in), FadeIn(img_gt, LEFT), Write(lbl_gt))

    # ÉVOLUTION AU CENTRE
    center_title = Text("DiffCMR Prediction", font_size=28, color=YELLOW).next_to(title, DOWN, buff=0.4)
    scene.play(Write(center_title))

    train_steps = [
        ("images/train/recon_b0_i1_10k.png", "10k", 0.1),
        ("images/train/recon_b0_i1 copy_20k.png", "20k", 0.2),
        ("images/train/recon_b0_i1_30K.png", "30k", 0.3),
        ("images/train/recon_b0_i1_150K.png", "150k", 1.0),
    ]

    center_pos = y_level
    current_img = None
    current_lbl = None
    current_progress = None
    
    # Barre de progression
    progress_bar_bg = Rectangle(width=4, height=0.1, color=GREY_E, fill_opacity=1).move_to(y_level + DOWN * 2.5)
    scene.play(ShowCreation(progress_bar_bg))

    for i, (path, text, progress) in enumerate(train_steps):
        new_img = ImageMobject(path).set_height(3).move_to(center_pos)
        
        is_final = (i == len(train_steps) - 1)
        lbl_color = TEAL if is_final else YELLOW
        lbl_text_str = f"Iteration: {text}" + (" (Converged)" if is_final else "")
        new_lbl = Text(lbl_text_str, font_size=20, color=lbl_color).next_to(new_img, DOWN, buff=0.3)
        
        # Remplissage de la barre
        progress_fill = Rectangle(width=4 * progress, height=0.1, color=lbl_color, fill_opacity=1, stroke_width=0)
        progress_fill.align_to(progress_bar_bg, LEFT).match_y(progress_bar_bg)

        if current_img is None:
            scene.play(FadeIn(new_img, UP), Write(new_lbl), FadeIn(progress_fill))
        else:
            scene.play(
                FadeOut(current_img), FadeOut(current_lbl),
                FadeIn(new_img), FadeIn(new_lbl),
                Transform(current_progress, progress_fill),
                run_time=1.0 
            )

        current_img = new_img
        current_lbl = new_lbl
        current_progress = progress_fill
        scene.wait(1)

    box_final = SurroundingRectangle(current_img, color=TEAL, buff=0.0)
    scene.play(ShowCreation(box_final), Flash(current_lbl, color=TEAL))
    
    scene.wait(1)
    scene.play(FadeOut(Group(*scene.mobjects)))




def play_quant_global_scene(scene):
    title = Text("From Diffusion to Flow Matching", color=TEAL).to_edge(UP)
    scene.play(Write(title))

    # --- INTRO VISUELLE ---
    img_diff = ImageMobject("images/eval/recon_b0_i1_diff.png").set_height(3)
    img_flow = ImageMobject("images/eval/recon_b0_i1_flow.png").set_height(3)
    img_gt = ImageMobject("images/eval/gt_b0_i1.png").set_height(3)
    images_group = Group(img_diff, img_flow, img_gt).arrange(RIGHT, buff=0.8).shift(DOWN * 0.5)

    lbl_diff = Text("DiffCMR Original", font_size=24, color=BLUE).next_to(img_diff, DOWN)
    lbl_flow = Text("Flow Matching (Ours)", font_size=28, color=YELLOW, weight=BOLD).next_to(img_flow, DOWN)
    lbl_gt = Text("Ground Truth", font_size=24, color=GREEN).next_to(img_gt, DOWN)

    scene.play(FadeIn(images_group, shift=UP), Write(VGroup(lbl_diff, lbl_flow, lbl_gt)))
    
    eq_text = Text("Visually Equivalent Quality...", font_size=32, color=LIGHT_BROWN).next_to(title, DOWN, buff=0.3)
    scene.play(Write(eq_text))
    scene.wait(2)

    scene.play(FadeOut(Group(images_group, lbl_diff, lbl_flow, lbl_gt, eq_text)))

    # --- ANALYSE GLOBALE (SPLIT SCREEN SÉCURISÉ) ---
    def_G = Text("1. Global Comparison: AccFactor 04 (CMRxRecon)", font_size=36, color=GOLD).next_to(title, DOWN, buff=0.3)
    scene.play(Write(def_G))

    # Tableau fixé à une largeur maximale pour éviter l'overlap
    img_G = ImageMobject("images/eval/global.png").set_width(7.5).to_edge(LEFT, buff=0.3).shift(DOWN * 0.5)
    scene.play(FadeIn(img_G, shift=RIGHT))

    # Panneau d'Analyse (Largeur contrôlée)
    analysis_bg = RoundedRectangle(width=5.5, height=4.5, corner_radius=0.2, color=GREY_E, fill_opacity=0.5).to_edge(RIGHT, buff=0.3).shift(DOWN * 0.5)
    a_title = Text("Performance Shift", font_size=28, color=WHITE).next_to(analysis_bg.get_top(), DOWN, buff=0.2)
    scene.play(FadeIn(analysis_bg), Write(a_title))

    # Surlignages
    hl_rep_G = Rectangle(width=img_G.get_width() + 0.1, height=0.3, color=RED, fill_opacity=0.3).move_to(img_G.get_bottom() + UP * 0.90)
    hl_our_G = Rectangle(width=img_G.get_width() + 0.1, height=0.3, color=GREEN, fill_opacity=0.3).move_to(img_G.get_bottom() + UP * 0.25)

    scene.play(ShowCreation(hl_rep_G), ShowCreation(hl_our_G))

    # Extraction Focus Temps & PSNR
    stat_time = Text("Time/img: 105s ➔ 4.3s", font_size=28, color=WHITE).move_to(analysis_bg.get_center() + UP * 1.0)
    time_badge = Text(" -96% Time ", font_size=32, color=BLACK, weight=BOLD).add_background_rectangle(color=GREEN, opacity=1, buff=0.1)
    time_badge.next_to(stat_time, DOWN, buff=0.2)

    stat_psnr = Text("PSNR: 37.13 ➔ 36.77", font_size=28, color=LIGHT_BROWN).next_to(time_badge, DOWN, buff=0.8)
    # On ajoute le pourcentage de baisse exact (-0.97%) dans le badge
    psnr_badge = Text(" Negligible PSNR Drop (-0.97%) ", font_size=24, color=YELLOW).next_to(stat_psnr, DOWN, buff=0.1)

    scene.play(Write(stat_time))
    scene.play(GrowFromCenter(time_badge))
    scene.wait(1.5)
    scene.play(Write(stat_psnr), Write(psnr_badge))
    scene.wait(1)

    scene.play(FadeOut(Group(*scene.mobjects)))




def play_quant_t_steps_scene(scene):
    title = Text("From Diffusion to Flow Matching", color=TEAL).to_edge(UP)
    def_T = Text("2. Integration Steps (T) & Variance Analysis", font_size=36, color=BLUE).next_to(title, DOWN, buff=0.3)

    # Écriture séquentielle
    scene.play(Write(title))
    scene.wait(0.2)
    scene.play(Write(def_T))

    # Layout anti-overlap
    img_T = ImageMobject("images/eval/test_T.png").set_width(7.5).to_edge(LEFT, buff=0.3).shift(DOWN * 0.5)
    scene.play(FadeIn(img_T, shift=RIGHT))

    analysis_bg = RoundedRectangle(width=5.5, height=4.5, corner_radius=0.2, color=GREY_E, fill_opacity=0.5).to_edge(RIGHT, buff=0.3).shift(DOWN * 0.5)
    a_title_T = Text("T-Dependence Analysis", font_size=28, color=WHITE).next_to(analysis_bg.get_top(), DOWN, buff=0.2)
    scene.play(FadeIn(analysis_bg), Write(a_title_T))

    # =========================================================
    # ÉTAPE 1 : LA FORTE VARIANCE DE LA DIFFUSION
    # =========================================================
    # On encadre tout le bloc Diffusion (à peu près la moitié haute)
    hl_diff_block = Rectangle(width=img_T.get_width() + 0.1, height=1.3, color=RED, fill_opacity=0.3).move_to(img_T.get_center() + UP * 0.75)
    scene.play(ShowCreation(hl_diff_block))

    obs_diff = Text("Diffusion: High Variance", font_size=28, color=RED).move_to(analysis_bg.get_center() + UP * 1.0)
    data_diff = Text("PSNR fluctuates heavily:\n25.5 (T=20) ➔ 37.1 (T=1000)", font_size=24, color=WHITE).next_to(obs_diff, DOWN, buff=0.2)
    conc_diff = Text("Requires huge T for fidelity.", font_size=22, color=LIGHT_BROWN).next_to(data_diff, DOWN, buff=0.2)

    scene.play(Write(obs_diff))
    scene.play(Write(data_diff))
    scene.play(FadeIn(conc_diff, shift=UP))
    scene.wait(2)

    # =========================================================
    # ÉTAPE 2 : LA STABILITÉ DU FLOW MATCHING
    # =========================================================
    scene.play(FadeOut(hl_diff_block), FadeOut(obs_diff), FadeOut(data_diff), FadeOut(conc_diff))

    # On encadre tout le bloc Flow Matching (moitié basse)
    hl_flow_block = Rectangle(width=img_T.get_width() + 0.1, height=1.7, color=GREEN, fill_opacity=0.2).move_to(img_T.get_center() + DOWN * 1.5)
    scene.play(ShowCreation(hl_flow_block))

    obs_flow = Text("Flow Matching: High Stability", font_size=28, color=GREEN).move_to(analysis_bg.get_center() + UP * 1.0)
    data_flow = Text("PSNR stays extremely stable:\n36.477 (T=1) ➔ 36.829 (T=4) ➔ 36.545 (T=100)", font_size=24, color=WHITE).next_to(obs_flow, DOWN, buff=0.2)
    conc_flow = Text("Robust performance at any step count.", font_size=22, color=LIGHT_BROWN).next_to(data_flow, DOWN, buff=0.2)

    scene.play(Write(obs_flow))
    scene.play(Write(data_flow))
    scene.play(FadeIn(conc_flow, shift=UP))
    scene.wait(2)

    # =========================================================
    # ÉTAPE 3 : LE PARADOXE DE L'OVERFITTING
    # =========================================================
    scene.play(FadeOut(hl_flow_block), FadeOut(obs_flow), FadeOut(data_flow), FadeOut(conc_flow))

    # On cible précisément les deux lignes à l'intérieur du bloc Flow Matching
    # T=4 (2ème ligne du bloc flow)
    hl_flow_4 = Rectangle(width=img_T.get_width() + 0.1, height=0.25, color=GREEN, fill_opacity=0.4).move_to(img_T.get_center() + DOWN * 0.85)
    # T=100 (Dernière ligne tout en bas) -> UP * 0.3 par rapport au bottom permet de remonter juste sur la ligne
    hl_flow_100 = Rectangle(width=img_T.get_width() + 0.1, height=0.25, color=ORANGE, fill_opacity=0.4).move_to(img_T.get_bottom() + UP * 0.47)

    paradox_title = Text("The Overfitting Paradox", font_size=28, color=YELLOW, weight=BOLD).move_to(analysis_bg.get_center() + UP * 1.0)
    scene.play(Write(paradox_title))

    # Apparition simultanée des deux cadres
    scene.play(ShowCreation(hl_flow_4), ShowCreation(hl_flow_100))

    paradox_data = Text("Optimal: T=4 (PSNR 36.82)\nDegraded: T=100 (PSNR 36.54)", font_size=24, color=WHITE, t2c={"T=100": ORANGE, "T=4": GREEN}).next_to(paradox_title, DOWN, buff=0.4)
    scene.play(Write(paradox_data))
    scene.wait(1)

    hyp_text = Text("Hypothesis:\nHigh T causes overfitting\non the CMR noise distribution.\nFlow works best on short paths.", font_size=22, color=LIGHT_BROWN, alignment="CENTER").next_to(paradox_data, DOWN, buff=0.5)

    scene.play(Write(hyp_text))
    scene.play(Write(hyp_text))
    scene.wait(1)

    scene.play(FadeOut(Group(*scene.mobjects)))




def play_quant_r_rounds_scene(scene):
    title = Text("From Diffusion to Flow Matching", color=TEAL).to_edge(UP)
    def_R = Text("3. Ensemble Averaging (R) & Stability", font_size=36, color=PURPLE_B).next_to(title, DOWN, buff=0.3)
    
    scene.play(Write(title))
    scene.play(Write(def_R))

    # Layout anti-overlap
    img_R = ImageMobject("images/eval/test_R.png").set_width(7.5).to_edge(LEFT, buff=0.3).shift(DOWN * 0.5)
    scene.play(FadeIn(img_R, shift=RIGHT))

    analysis_bg = RoundedRectangle(width=5.5, height=4.5, corner_radius=0.2, color=GREY_E, fill_opacity=0.5).to_edge(RIGHT, buff=0.3).shift(DOWN * 0.5)
    a_title_R = Text("PSNR Stability Analysis", font_size=28, color=WHITE).next_to(analysis_bg.get_top(), DOWN, buff=0.2)
    # On laisse groupé (Fond + Texte)
    scene.play(FadeIn(analysis_bg), Write(a_title_R))

    # --- Phase Diffusion (Bleu) ---
    hl_R_diff = Rectangle(width=img_R.get_width() + 0.1, height=1.3, color=BLUE, fill_opacity=0.3).move_to(img_R.get_center() + UP * 0.3)

    scene.play(ShowCreation(hl_R_diff))
    r_obs = Text("Diffusion R-Dependence:", font_size=26, color=WHITE).move_to(analysis_bg.get_center() + UP * 0.8)

    # Focus uniquement sur le PSNR
    r_data_d = Text("R=1: PSNR 33.89\nR=8: PSNR 36.68", font_size=26, color=BLUE).next_to(r_obs, DOWN, buff=0.2)
    r_diff_badge = Text(" +2.8 dB Jump! ", font_size=24, color=WHITE).add_background_rectangle(color=BLUE, opacity=1).next_to(r_data_d, RIGHT, buff=0.2)

    scene.play(Write(r_obs))
    scene.play(Write(r_data_d), GrowFromCenter(r_diff_badge))
    
    r_time_d = Text("Highly unstable at low R.", font_size=22, color=LIGHT_BROWN, alignment="CENTER").next_to(r_data_d, DOWN, buff=0.3)
    scene.play(Write(r_time_d))
    scene.wait(1)

    # --- Phase Flow Matching (Violet) ---
    # On laisse groupé
    scene.play(FadeOut(hl_R_diff), FadeOut(r_obs), FadeOut(r_data_d), FadeOut(r_diff_badge), FadeOut(r_time_d))

    hl_R_flow = Rectangle(width=img_R.get_width() + 0.1, height=1.0, color=PURPLE_B, fill_opacity=0.3).move_to(img_R.get_bottom() + UP * 0.8)
    scene.play(ShowCreation(hl_R_flow))

    r_stab = Text("Flow Matching Stability:", font_size=26, color=WHITE).move_to(analysis_bg.get_center() + UP * 0.8)
    r_data_f = Text("R=1: PSNR 36.35\nR=8: PSNR 36.83", font_size=26, color=PURPLE_A).next_to(r_stab, DOWN, buff=0.2)
    r_flow_badge = Text(" +0.4 dB Stable ", font_size=24, color=WHITE).add_background_rectangle(color=PURPLE_B, opacity=1).next_to(r_data_f, RIGHT, buff=0.2)

    conc_text = Text("High fidelity achieved immediately.\nNo need for heavy averaging.", font_size=22, color=LIGHT_BROWN, alignment="CENTER").next_to(r_data_f, DOWN, buff=0.4)

    scene.play(Write(r_stab))
    scene.play(Write(r_data_f), GrowFromCenter(r_flow_badge))
    scene.wait(1)

    # On laisse groupé (Texte
    scene.play(Write(conc_text))
    scene.wait(1)

    scene.play(FadeOut(Group(*scene.mobjects)))
