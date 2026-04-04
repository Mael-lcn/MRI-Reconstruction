from manimlib import *



def play_intro_scene(scene):
    """Fonction principale de l'intro, appelée par main.py"""
    scene.wait(1)

    show_paper_first_page(scene)

    # On récupère les objets générés pour pouvoir les réutiliser dans le reverse process
    img_0, img_t, dots, img_T = show_forward_process(scene)
    
    show_reverse_process(scene, img_0, img_t, dots, img_T)
    cleanup(scene)


# ==========================================
# SOUS-MÉTHODES DE L'INTRODUCTION
# ==========================================
def show_paper_first_page(scene):
    # Remplacer par le chemin vers ton image du papier
    paper_img = ImageMobject("images/DDPM_paper.png").set_height(6.0)

    scene.play(FadeIn(paper_img, UP), run_time=1)
    scene.wait(2)
    scene.play(FadeOut(paper_img, DOWN), run_time=0.8)
    scene.wait(0.5)


def show_forward_process(scene):
    # --- 1. Images et séquence ---
    img_0 = ImageMobject("images/diffusion/image_0.png").set_height(2.0)
    img_t = ImageMobject("images/diffusion/image_t.png").set_height(2.0)
    dots = Tex(r"\dots", font_size=60)
    img_T = ImageMobject("images/diffusion/image_tf.png").set_height(2.0)

    sequence = Group(img_0, img_t, dots, img_T)
    sequence.arrange(RIGHT, buff=1.5)
    sequence.shift(UP * 0.5)

    # --- 2. Variables (x0, xt, xT) ---
    label_0 = Tex("x_0").next_to(img_0, DOWN, buff=0.4)
    label_t = Tex("x_t").next_to(img_t, DOWN, buff=0.4)
    label_T = Tex("x_T").next_to(img_T, DOWN, buff=0.4)

    noise_text = Text("Pure Noise", font_size=20, color=GREY_B).next_to(label_T, DOWN, buff=0.2)

    # --- 3. Flèches Forward et équation ---
    arrow_f1 = Arrow(img_0.get_right(), img_t.get_left(), buff=0.2, color=RED)
    arrow_f2 = Arrow(img_t.get_right(), dots.get_left(), buff=0.2, color=RED)
    arrow_f3 = Arrow(dots.get_right(), img_T.get_left(), buff=0.2, color=RED)

    q1 = Tex("q", font_size=32, color=RED).next_to(arrow_f1, UP, buff=0.1)
    q2 = Tex("q", font_size=32, color=RED).next_to(arrow_f2, UP, buff=0.1)
    q3 = Tex("q", font_size=32, color=RED).next_to(arrow_f3, UP, buff=0.1)

    q_main = Tex("q(x_t | x_{t-1})", font_size=56, color=RED).to_edge(UP, buff=0.5)

    # --- 4. Animation ---
    scene.play(FadeIn(img_0), Write(label_0), run_time=0.5)
    scene.play(GrowArrow(arrow_f1), Write(q1), run_time=0.4)
    scene.play(FadeIn(img_t), Write(label_t), Write(q_main), run_time=0.5)
    scene.play(GrowArrow(arrow_f2), Write(q2), FadeIn(dots), run_time=0.4)
    scene.play(GrowArrow(arrow_f3), Write(q3), run_time=0.4)
    scene.play(FadeIn(img_T), Write(label_T), Write(noise_text), run_time=0.5)
    scene.wait(1)

    # On retourne ces objets car la méthode reverse en a besoin pour s'aligner
    return img_0, img_t, dots, img_T


def show_reverse_process(scene, img_0, img_t, dots, img_T):
    # --- 1. Ligne d'ancrage stricte ---
    marge_basse = DOWN * 1.2
    
    anchor_T = img_T.get_bottom() + marge_basse
    anchor_dots = dots.get_bottom() + marge_basse
    anchor_t = img_t.get_bottom() + marge_basse
    anchor_0 = img_0.get_bottom() + marge_basse

    # --- 2. Flèches courbes (Reverse) ---
    arrow_r1 = CurvedArrow(anchor_T, anchor_dots, angle=TAU/4, color=BLUE)
    arrow_r2 = CurvedArrow(anchor_dots, anchor_t, angle=TAU/4, color=BLUE)
    arrow_r3 = CurvedArrow(anchor_t, anchor_0, angle=TAU/4, color=BLUE)

    p1 = Tex("p_\\theta", font_size=32, color=BLUE).next_to(arrow_r1, DOWN, buff=0.1)
    p2 = Tex("p_\\theta", font_size=32, color=BLUE).next_to(arrow_r2, DOWN, buff=0.1)
    p3 = Tex("p_\\theta", font_size=32, color=BLUE).next_to(arrow_r3, DOWN, buff=0.1)

    p_main = Tex("p_\\theta(x_{t-1} | x_t)", font_size=56, color=BLUE).to_edge(DOWN, buff=0.5)

    # --- 3. Animation ---
    scene.play(GrowArrow(arrow_r1), Write(p1), run_time=0.4)
    scene.play(GrowArrow(arrow_r2), Write(p2), Write(p_main), run_time=0.4)
    scene.play(GrowArrow(arrow_r3), Write(p3), run_time=0.4)
    scene.wait(2)


def cleanup(scene):
    # Nettoie la scène pour préparer le terrain à main.py pour la partie 2
    scene.play(FadeOut(Group(*scene.mobjects)), run_time=1)
