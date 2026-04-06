import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manimlib import *


from utils import cleanup



# ==========================================
# 1. INTRO & PAPER FLIP
# ==========================================
def play_intro_scene(scene):
    scene.wait(0.5)

    # 1. Titre principal du projet (DiffCMR)
    title_l1 = Text("DiffCMR: Fast Cardiac MRI Reconstruction", font_size=44, weight=BOLD)
    title_l2 = Text("with Diffusion Probabilistic Models", font_size=44, weight=BOLD)
    
    title_group = VGroup(title_l1, title_l2).arrange(DOWN, buff=0.3)
    
    scene.play(Write(title_group), run_time=1.5)
    scene.wait(1)

    # 2. Transition : La question
    question = Text("-> What is diffusion?", font_size=38, color=BLUE)
    question.next_to(title_group, DOWN, buff=1.2)

    scene.play(FadeIn(question, UP), run_time=0.8)
    scene.wait(1.5)

    # On nettoie l'écran avant de faire apparaître le papier
    scene.play(FadeOut(title_group), FadeOut(question), run_time=0.8)
    scene.wait(0.5)

    # 3. Page de Garde du papier fondateur (DDPM)
    front_page = ImageMobject("images/DDPM_paper.png")
    front_page.set_height(6.0) 

    scene.play(FadeIn(front_page, UP), run_time=1)
    scene.wait(1.5)
    
    # 4. Effet de "Feuilletage"
    scene.play(front_page.animate.shift(UP * 10), run_time=0.8, rate_func=rush_into)

    # 5. Le schéma de base (Forward)
    img_0, img_t, dots, img_T = show_forward_process(scene)

    # 6. Le modèle de Deep Learning (Reverse)
    show_reverse_process(scene, img_0, img_t, dots, img_T)
    
    scene.wait(3)
    cleanup(scene)


def show_forward_process(scene):
    # Chargement des images
    img_0 = ImageMobject("images/diffusion/image_0.png").set_height(2.0)
    img_t = ImageMobject("images/diffusion/image_t.png").set_height(2.0)
    dots = Tex(r"\dots", font_size=60)
    img_T = ImageMobject("images/diffusion/image_tf.png").set_height(2.0)

    # Alignement central (légèrement remonté pour laisser la place aux flèches bleues en bas)
    sequence = Group(img_0, dots, img_t, img_T).arrange(RIGHT, buff=1.2).shift(UP * 0.5)

    # Labels sous les images
    label_0 = Tex(r"x_0").next_to(img_0, DOWN, buff=0.4)
    label_t = Tex(r"x_t").next_to(img_t, DOWN, buff=0.4)
    label_T = Tex(r"x_T").next_to(img_T, DOWN, buff=0.4)

    # Flèches d'ajout de bruit (Rouges)
    arrow_f1 = Arrow(img_0.get_right(), dots.get_left(), buff=0.2, color=RED)
    arrow_f2 = Arrow(dots.get_right(), img_t.get_left(), buff=0.2, color=RED)
    arrow_f3 = Arrow(img_t.get_right(), img_T.get_left(), buff=0.2, color=RED)

    # Titre du processus Forward (En haut)
    q_main = Tex(r"q(x_t | x_{t-1})", font_size=56, color=RED).to_edge(UP, buff=0.2)
    q_text = Text("Forward Process: Adding Gaussian Noise", font_size=28, color=RED).next_to(q_main, DOWN, buff=0.2)

    scene.play(FadeIn(img_0), Write(label_0), run_time=0.5)
    scene.play(Write(q_main), Write(q_text))
    scene.play(GrowArrow(arrow_f1), FadeIn(dots), GrowArrow(arrow_f2), run_time=0.5)
    scene.play(FadeIn(img_t), Write(label_t), run_time=0.5)
    scene.play(GrowArrow(arrow_f3), run_time=0.5)
    scene.play(FadeIn(img_T), Write(label_T), run_time=0.5)
    scene.wait(1)
    
    return img_0, img_t, dots, img_T


def show_reverse_process(scene, img_0, img_t, dots, img_T):
    # On calcule une ligne d'ancrage bien en dessous des labels x_0, x_t...
    y_level = img_0.get_bottom()[1] - 1.2 

    # Coordonnées X alignées avec le centre de chaque image
    anchor_T = np.array([img_T.get_center()[0], y_level, 0])
    anchor_t = np.array([img_t.get_center()[0], y_level, 0])
    anchor_dots = np.array([dots.get_center()[0], y_level, 0])
    anchor_0 = np.array([img_0.get_center()[0], y_level, 0])

    # Flèches courbes (angle=-TAU/4 fait courber vers le bas quand on va de droite à gauche)
    arrow_r1 = CurvedArrow(anchor_T, anchor_t, angle=-TAU/4, color=BLUE)
    arrow_r2 = CurvedArrow(anchor_t, anchor_dots, angle=-TAU/4, color=BLUE)
    arrow_r3 = CurvedArrow(anchor_dots, anchor_0, angle=-TAU/4, color=BLUE)

    # L'équation du modèle et son explication (En bas)
    p_main = Tex(r"p_\theta(x_{t-1} | x_t)", font_size=56, color=BLUE).to_edge(DOWN, buff=0.3)
    p_text = Text("Reverse Process: Neural Network Denoising", font_size=28, color=BLUE).next_to(p_main, UP, buff=0.2)

    scene.play(Write(p_main), Write(p_text))

    # On anime les flèches de droite à gauche pour mimer la reconstruction
    scene.play(GrowArrow(arrow_r1), run_time=0.8)
    scene.play(GrowArrow(arrow_r2), GrowArrow(arrow_r3), run_time=1.2)




# ==========================================
# 2. FORWARD
# ==========================================
def play_forward_math_scene(scene):
    # --- 1. TITRE INITIAL ---
    title = Text("1. The Forward Process: Rigorous Derivation", color=RED).to_edge(UP)
    scene.play(Write(title))
    scene.wait(1)

    # PAGE 1 : RÉCURSION
    eq_alpha = Tex(
        r"\alpha_t := 1 - \beta_t \quad \text{and} \quad \bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s", 
        font_size=40
    ).next_to(title, DOWN, buff=0.5)

    step_txt = Text("Reparameterization at step t:", font_size=28, color=GREY_A).next_to(eq_alpha, DOWN, buff=0.6)
    eq_step_t = Tex(r"x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t", font_size=42).next_to(step_txt, DOWN, buff=0.3)

    sub_txt = Tex(r"\text{Substitute } x_{t-1} \text{ to show the recursion:}", font_size=32, color=YELLOW).next_to(eq_step_t, DOWN, buff=0.6)
    eq_sub = Tex(
        r"x_t = \sqrt{\alpha_t} ( \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-1} ) + \sqrt{1 - \alpha_t} \epsilon_t", 
        font_size=34
    ).next_to(sub_txt, DOWN, buff=0.3)

    scene.play(FadeIn(eq_alpha, UP))
    scene.play(Write(step_txt), FadeIn(eq_step_t, UP))
    scene.wait(1)
    scene.play(Write(sub_txt), FadeIn(eq_sub, UP))
    scene.wait(2)

    # PAGE 2 : FUSION & CLOSED FORM
    scene.play(
        FadeOut(eq_alpha), FadeOut(step_txt), FadeOut(eq_step_t),
        VGroup(sub_txt, eq_sub).animate.next_to(title, DOWN, buff=0.5)
    )

    merge_txt = Tex(
        r"\text{Summing independent noises: } \mathcal{N}(0, \sigma_1^2) + \mathcal{N}(0, \sigma_2^2) = \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)", 
        font_size=30, color=GREEN
    ).next_to(eq_sub, DOWN, buff=0.6)
    
    eq_merged = Tex(r"x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\epsilon}_{t-2}", font_size=42).next_to(merge_txt, DOWN, buff=0.3)

    scene.play(Write(merge_txt), FadeIn(eq_merged, UP))
    scene.wait(2)

    general_txt = Tex(r"\text{Closed Form (unrolled to } x_0 \text{):}", font_size=32, color=YELLOW).next_to(eq_merged, DOWN, buff=0.8)
    eq_closed = Tex(r"x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon", font_size=48).next_to(general_txt, DOWN, buff=0.3)
    
    box_closed = SurroundingRectangle(eq_closed, color=RED_B, buff=0.2)

    scene.play(Write(general_txt))
    scene.play(FadeIn(eq_closed, UP), ShowCreation(box_closed))
    scene.wait(3)

    # --- NETTOYAGE TOTAL ---
    scene.play(FadeOut(Group(*scene.mobjects)), run_time=1)
    scene.wait(0.5)

    # --- APPEL BROWNIEN ---
    play_brownian_trajectory_scene(scene)


def play_brownian_trajectory_scene(scene):
    # Nouveau titre pour cette scène propre
    subtitle = Text("High-Dimensional Brownian Trajectory", font_size=36, color=BLUE).to_edge(UP)
    scene.play(Write(subtitle))

    # 1. Image x0
    img_0 = ImageMobject("images/diffusion/image_0.png").set_height(2.5).move_to(LEFT * 4.5)
    
    label_img = Tex(r"x_0 \in \mathbb{R}^{128 \times 128}", font_size=32).next_to(img_0, DOWN)

    scene.play(FadeIn(img_0), Write(label_img))
    scene.wait(1)

    # 2. Espace de phase
    axes = ThreeDAxes(width=5, height=5, depth=5).move_to(RIGHT * 2)
    start_point = axes.c2p(-1.5, -1.5, -1)
    dot_start = Dot(start_point, color=WHITE, radius=0.06)
    label_x0 = Tex(r"x_0", font_size=36).next_to(dot_start, LEFT, buff=0.2)

    scene.play(ShowCreation(axes))
    
    # On anime l'image qui se réduit vers le point
    scene.play(
        img_0.animate.scale(0.1).move_to(start_point).set_opacity(0),
        FadeIn(dot_start),
        Write(label_x0),
        run_time=1.5
    )
    scene.remove(img_0) # On nettoie l'image devenue invisible
    scene.wait(0.5)

    # 3. Génération du Zig-Zag jaune
    num_steps = 30
    np.random.seed(15)
    points = [start_point]
    curr = start_point
    for _ in range(num_steps):
        # On ajoute un vecteur de bruit aléatoire
        curr = curr + np.random.normal(0, 0.22, 3)
        points.append(curr)

    trajectory = VMobject(color=YELLOW, stroke_width=2)
    trajectory.set_points_as_corners(points) # L'effet Zig-Zag

    scene.play(ShowCreation(trajectory), run_time=5, rate_func=linear)

    dot_end = Dot(points[-1], color=RED, radius=0.1) # Radius augmenté (0.06 -> 0.1)
    label_end = Tex(r"x_T", font_size=56).next_to(dot_end, UR, buff=0.15) # Font size massif (32 -> 56)

    scene.play(FadeIn(dot_end), Write(label_end))
    scene.wait(1)

    # Nettoyage final
    scene.play(FadeOut(Group(*scene.mobjects)))





# ==========================================
# 3. REVERSE DIFFUSION
# ==========================================
def play_reverse_math_scene(scene):
    # --- 1. TITRE ---
    title = Text("2. The Reverse Process: Generative Denoising", color=BLUE).to_edge(UP)
    scene.play(Write(title))
    scene.wait(1)

    # =========================================================
    # PAGE 1 : L'INTRACTABILITÉ
    # =========================================================
    p_text = Text("The true posterior is unknown and requires the entire dataset:", font_size=28, color=GREY_A)
    p_text.next_to(title, DOWN, buff=0.5)

    # On simplifie la formule pour le parsing
    eq_q_reverse = Tex(
        r"q(x_{t-1} | x_t) = \int q(x_{t-1} | x_0, x_t) q(x_0 | x_t) dx_0", 
        font_size=40
    ).next_to(p_text, DOWN, buff=0.5)

    warning = Text("Intractable in practice!", font_size=32, color=RED).next_to(eq_q_reverse, DOWN, buff=0.6)

    scene.play(Write(p_text), FadeIn(eq_q_reverse, UP))
    scene.play(Write(warning))
    scene.wait(2)

    # NETTOYAGE TOTAL PAGE 1
    # On utilise clear() pour être certain qu'il ne reste rien en mémoire
    scene.play(FadeOut(VGroup(p_text, eq_q_reverse, warning)))
    scene.clear() 
    scene.add(title) # On remet le titre qui a été effacé par clear()

    # =========================================================
    # PAGE 2 : BAYES & CONDITIONNEMENT
    # =========================================================
    bayes_text = Text("But conditioned on x0, the reverse step becomes tractable:", font_size=28, color=YELLOW)
    bayes_text.next_to(title, DOWN, buff=0.5)
    
    eq_bayes = Tex(
        r"q(x_{t-1} | x_t, x_0) = q(x_t | x_{t-1}, x_0) \frac{q(x_{t-1} | x_0)}{q(x_t | x_0)}", 
        font_size=42
    ).next_to(bayes_text, DOWN, buff=0.6)

    scene.play(Write(bayes_text))
    scene.play(FadeIn(eq_bayes, UP))
    scene.wait(2)

    # On ajoute la forme Gaussienne en dessous
    tract_text = Text("This leads to a closed-form Gaussian distribution:", font_size=28, color=GREY_A)
    tract_text.next_to(eq_bayes, DOWN, buff=0.8)

    eq_tractable = Tex(
        r"q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})", 
        font_size=42
    ).next_to(tract_text, DOWN, buff=0.4)

    scene.play(Write(tract_text), FadeIn(eq_tractable, UP))
    scene.wait(2)

    # NETTOYAGE TOTAL PAGE 2
    scene.play(FadeOut(VGroup(bayes_text, eq_bayes, tract_text, eq_tractable)))
    scene.clear()
    scene.add(title)

    # =========================================================
    # PAGE 3 : LES PARAMÈTRES CIBLES
    # =========================================================
    params_title = Text("Exact Mean and Variance (Ho et al. 2020):", font_size=32, color=YELLOW)
    params_title.next_to(title, DOWN, buff=0.5)

    # On découpe la moyenne en deux lignes
    eq_mu_tilde = Tex(
        r"\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t", 
        font_size=36
    ).next_to(params_title, DOWN, buff=0.8)

    eq_beta_tilde = Tex(
        r"\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t", 
        font_size=40
    ).next_to(eq_mu_tilde, DOWN, buff=0.8)

    scene.play(Write(params_title))
    scene.play(FadeIn(eq_mu_tilde, UP))
    scene.play(FadeIn(eq_beta_tilde, UP))
    scene.wait(2)

    # NETTOYAGE TOTAL PAGE 3
    scene.play(FadeOut(VGroup(params_title, eq_mu_tilde, eq_beta_tilde)))
    scene.clear()
    scene.add(title)  
    cleanup(scene)
    scene.wait(1)




# ==========================================
# 4. THE LOSS FUNCTION
# ==========================================
def play_loss_scene(scene):
    # --- 1. TITRE ---
    title = Text("3. The Objective: Deriving the Simple Loss", color=PURPLE).to_edge(UP)
    scene.play(Write(title))

    # =========================================================
    # PAGE 1 : LE D_KL ENTRE DEUX GAUSSIENNES
    # =========================================================
    kl_text = Text("When variance is fixed to a constant Sigma,", font_size=28, color=GREY_A)
    kl_text.next_to(title, DOWN, buff=0.5)

    # Propriété mathématique fondamentale du D_KL pour des Gaussiennes
    eq_kl_prop = Tex(
        r"D_{KL}(\mathcal{N}(\mu_1, \sigma^2 \mathbf{I}) \parallel \mathcal{N}(\mu_2, \sigma^2 \mathbf{I})) = \frac{1}{2\sigma^2} \left\Vert \mu_1 - \mu_2 \right\Vert^2",
        font_size=40
    ).next_to(kl_text, DOWN, buff=0.6)

    scene.play(Write(kl_text), FadeIn(eq_kl_prop, UP))
    scene.wait(2)

    # Application au cas DDPM
    app_text = Text("Applied to our reverse step:", font_size=28, color=YELLOW).next_to(eq_kl_prop, DOWN, buff=0.8)
    eq_lt = Tex(
        r"L_t = \mathbb{E}_q \left[ \frac{1}{2\sigma_t^2} \left\Vert \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \right\Vert^2 \right]",
        font_size=42
    ).next_to(app_text, DOWN, buff=0.4)

    scene.play(Write(app_text), FadeIn(eq_lt, UP))
    scene.wait(2.5)

    # NETTOYAGE PAGE 1
    scene.play(FadeOut(VGroup(kl_text, eq_kl_prop, app_text, eq_lt)))
    scene.clear()
    scene.add(title)

    # =========================================================
    # PAGE 2 : SUBSTITUTION DE MU_TILDE (LA CIBLE)
    # =========================================================
    sub_title = Text("Recall the target mean from the forward process:", font_size=32, color=YELLOW)
    sub_title.next_to(title, DOWN, buff=0.5)

    # On rappelle que mu_tilde s'écrit en fonction de x_t et epsilon
    eq_mu_target = Tex(
        r"\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)",
        font_size=42
    ).next_to(sub_title, DOWN, buff=0.8)
    
    note_eps = Tex(r"\text{where } \epsilon \text{ is the actual noise added to } x_0", font_size=28, color=GREY_A).next_to(eq_mu_target, DOWN, buff=0.4)

    scene.play(Write(sub_title))
    scene.play(FadeIn(eq_mu_target, UP), Write(note_eps))
    scene.wait(2.5)

    # NETTOYAGE PAGE 2
    scene.play(FadeOut(VGroup(sub_title, eq_mu_target, note_eps)))
    scene.clear()
    scene.add(title)

    # =========================================================
    # PAGE 3 : PARAMÉTRISATION DU MODÈLE (MU_THETA)
    # =========================================================
    model_text = Text("We choose to parameterize our model similarly:", font_size=32, color=BLUE)
    model_text.next_to(title, DOWN, buff=0.5)

    eq_mu_theta = Tex(
        r"\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)",
        font_size=42
    ).next_to(model_text, DOWN, buff=0.8)

    scene.play(Write(model_text))
    scene.play(FadeIn(eq_mu_theta, UP))
    scene.wait(2)

    # NETTOYAGE PAGE 3
    scene.play(FadeOut(VGroup(model_text, eq_mu_theta)))
    scene.clear()
    scene.add(title)

    # =========================================================
    # PAGE 4 : SIMPLIFICATION FINALE (MSE)
    # =========================================================
    final_text = Tex(r"\text{Subtracting the two means, the } x_{t} \text{ terms cancel out:}", font_size=30, color=GREY_A)
    final_text.next_to(title, DOWN, buff=0.5)

    # On montre la soustraction qui mène à epsilon - epsilon_theta
    eq_cancel = Tex(
        r"L_t \propto \left\Vert \frac{\beta_t}{\sqrt{\alpha_t(1-\bar{\alpha}_t)}} ( \epsilon - \epsilon_\theta(x_t, t) ) \right\Vert^2",
        font_size=40
    ).next_to(final_text, DOWN, buff=0.8)

    scene.play(Write(final_text), FadeIn(eq_cancel, UP))
    scene.wait(2)

    # La Simplified Loss (Ho et al. 2020)
    ho_text = Text("By ignoring the weighting terms, we get the Simple Loss:", font_size=34, color=GREEN)
    ho_text.next_to(eq_cancel, DOWN, buff=1)

    eq_loss_simple = Tex(
        r"L_{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\Vert \epsilon - \epsilon_\theta(x_t, t) \right\Vert^2 \right]",
        font_size=52
    ).next_to(ho_text, DOWN, buff=0.5)

    box_final = SurroundingRectangle(eq_loss_simple, color=GREEN, buff=0.2)

    scene.play(Write(ho_text))
    scene.play(FadeIn(eq_loss_simple, UP))
    scene.play(ShowCreation(box_final))
    
    scene.wait(1)
    cleanup(scene)
