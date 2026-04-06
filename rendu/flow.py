import random

from manimlib import *



def play_flow_matching_euler_scene(scene):
    # --- 1. TITRE ---
    title = Text("Flow Matching: Mathematical Framework", color=GOLD).to_edge(UP, buff=0.3)
    scene.play(Write(title))

    # ==========================================
    # PARTIE 1 : DÉFINITIONS MATHÉMATIQUES STRICTES
    # ==========================================

    # ------------------------------------------
    # ÉCRAN A : L'ODE Continue et ses variables
    # ------------------------------------------
    step1 = Text("1. Probability Flow ODE", font_size=32, color=BLUE_B).move_to(UP * 1.8)
    ode_form = Tex(r"\frac{d X_t}{dt} = v_\theta(X_t, t)", font_size=52).next_to(step1, DOWN, buff=0.5)

    # Définition stricte de chaque terme
    defs_ode = VGroup(
        Tex(r"\bullet \,\, t \in [0, 1] : \text{Continuous time variable}", font_size=28, color=GREY),
        Tex(r"\bullet \,\, X_t \in \mathbb{R}^{H \times W} : \text{Image state at time } t", font_size=28, color=GREY),
        Tex(r"\bullet \,\, v_\theta : \text{Neural Network predicting the vector field}", font_size=28, color=GREY)
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(ode_form, DOWN, buff=0.6)

    scene.play(Write(step1), run_time=0.8)
    scene.play(FadeIn(ode_form, DOWN), run_time=0.8)
    scene.play(Write(defs_ode), run_time=1.5)
    scene.wait(2.5)

    # ------------------------------------------
    # ÉCRAN B : Conditions aux limites & Target
    # ------------------------------------------
    step2 = Text("2. Boundary Conditions & Training Target", font_size=32, color=GREEN_B).move_to(step1)
    
    # Définition des bornes
    bounds = Tex(r"X_0 \sim \mathcal{N}(0, I) \quad \rightarrow \quad X_1 \sim p_{data}", font_size=40).next_to(step2, DOWN, buff=0.5)
    
    # Définition de la cible d'entraînement
    target_form = Tex(r"v_{target}(X_t) = X_1 - X_0", font_size=52, color=GREEN_B).next_to(bounds, DOWN, buff=0.5)
    def_target = Tex(r"\text{The target is a constant velocity straight path}", font_size=28, color=GREY).next_to(target_form, DOWN, buff=0.3)
    
    scene.play(
        ReplacementTransform(step1, step2),
        ReplacementTransform(ode_form, bounds),
        ReplacementTransform(defs_ode, target_form),
        run_time=1.2
    )
    scene.play(Write(def_target), run_time=0.8)
    scene.wait(2.5)

    # ------------------------------------------
    # ÉCRAN C : Intégration d'Euler et Discrétisation
    # ------------------------------------------
    step3 = Text("3. Discrete Euler Sampling (Inference)", font_size=32, color=YELLOW).move_to(step2)
    euler_form = Tex(r"X_{t+\Delta t} = X_t + v_\theta(X_t, t) \cdot \Delta t", font_size=52, color=YELLOW).next_to(step3, DOWN, buff=0.5)
    
    # Définition stricte du pas de temps
    def_dt = Tex(r"\Delta t = \frac{1}{N} \quad \text{where usually } N \in [20, 40] \text{ steps}", font_size=32, color=GREY).next_to(euler_form, DOWN, buff=0.5)    

    scene.play(
        ReplacementTransform(step2, step3),
        ReplacementTransform(bounds, euler_form),
        FadeOut(target_form),
        ReplacementTransform(def_target, def_dt),
        run_time=1.2
    )
    scene.wait(2.5)

    # ==========================================
    # TRANSITION : THÉORIE -> VISUEL
    # ==========================================
    scene.play(
        FadeOut(step3),
        FadeOut(def_dt),
        euler_form.animate.scale(0.7).next_to(title, DOWN, buff=0.3).set_color(WHITE),
        run_time=1
    )

    # ==========================================
    # PARTIE 2 : VISUALISATION DYNAMIQUE
    # ==========================================
    BOX_SIZE = 1.6
    pos_start = LEFT * 4.5 + DOWN * 1.5
    pos_end = RIGHT * 4.5 + DOWN * 1.5

    def get_state(t_val):
        current_pos = pos_start * (1 - t_val) + pos_end * t_val
        frame = Square(side_length=BOX_SIZE, color=WHITE, stroke_width=2)

        myo = Circle(radius=0.45, color=GREY, fill_opacity=0.6 * t_val, stroke_width=0)
        vent = Circle(radius=0.2, color=BLACK, fill_opacity=1.0 * t_val, stroke_width=0)
        clean_img = VGroup(myo, vent)

        noise = VGroup()
        random.seed(99) 
        for _ in range(250):
            x = random.uniform(-BOX_SIZE/2.1, BOX_SIZE/2.1)
            y = random.uniform(-BOX_SIZE/2.1, BOX_SIZE/2.1)
            noise.add(Dot(radius=0.02, color=WHITE, fill_opacity=1.0 * (1 - t_val)).move_to([x, y, 0]))

        return VGroup(frame, clean_img, noise).move_to(current_pos)

    path_line = DashedLine(pos_start, pos_end, color=GREY, dash_length=0.15)
    scene.play(ShowCreation(path_line), run_time=0.8)

    steps = 2  
    dt = 1.0 / steps
    current_t = 0.0
    
    current_img = get_state(current_t)
    lbl_state = Tex(f"X_{{{current_t:.2f}}}", font_size=32).next_to(current_img, DOWN, buff=0.3)
    lbl_noise = Text("Pure Noise", font_size=24, color=RED_B).next_to(lbl_state, DOWN, buff=0.2)

    scene.play(FadeIn(current_img, scale=0.5), Write(lbl_state), Write(lbl_noise), run_time=1)
    scene.wait(0.5)
    scene.play(FadeOut(lbl_noise), run_time=0.5)

    for step in range(steps):
        next_t = current_t + dt
        
        start_pt = current_img.get_right()
        end_pt = get_state(next_t).get_left()
        
        v_arrow = Arrow(start_pt, start_pt + RIGHT * (end_pt[0] - start_pt[0]), color=RED_B, buff=0)
        v_lbl = Tex(r"v_\theta \cdot \Delta t", font_size=28, color=RED_B).next_to(v_arrow, UP, buff=0.1)

        scene.play(
            GrowArrow(v_arrow), 
            Write(v_lbl),
            euler_form.animate.set_color_by_tex("v_\\theta", RED_B),
            run_time=0.7
        )
        scene.wait(0.3) 

        next_img = get_state(next_t)
        next_lbl = Tex(f"X_{{{next_t:.2f}}}", font_size=32).next_to(next_img, DOWN, buff=0.3)
        
        scene.play(
            ReplacementTransform(current_img, next_img),
            ReplacementTransform(lbl_state, next_lbl),
            FadeOut(v_arrow),
            FadeOut(v_lbl),
            euler_form.animate.set_color(WHITE), 
            run_time=1.0 
        )
        
        current_img = next_img
        lbl_state = next_lbl
        current_t = next_t
        scene.wait(0.3)

    final_lbl = Text("Target Reached", font_size=24, color=GREEN_B).next_to(lbl_state, DOWN, buff=0.2)
    scene.play(Write(final_lbl), run_time=0.6)
    
    scene.play(
        current_img[1].animate.set_color(GREEN_B),
        Indicate(current_img, color=GREEN_B, scale_factor=1.05),
        run_time=1.0
    )
    scene.wait(2)

    scene.play(FadeOut(Group(*scene.mobjects)), run_time=1)
