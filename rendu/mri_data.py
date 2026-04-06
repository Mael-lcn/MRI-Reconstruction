from manimlib import *
import numpy as np



def play_ifft_rss_scene(scene):
    # --- 1. TITRE ET LAYOUT STRICT ---
    # On réserve le haut de l'écran (y > 1.5) pour les explications
    # On réserve le bas de l'écran (y < 1.5) pour les schémas
    title = Text("Step 2: Spatial Reconstruction (Single Frame)", color=GOLD).to_edge(UP, buff=0.2)
    scene.play(Write(title))

    # --- 2. FONCTIONS DE DESSIN (Tailles fixes) ---
    BOX_SIZE = 1.0 # Taille réduite pour garantir qu'aucune des 4 lignes ne déborde

    def get_kspace_mockup(color=WHITE):
        frame = Square(side_length=BOX_SIZE, color=GREY, stroke_width=2)
        dots = VGroup()
        for x in np.linspace(-0.4, 0.4, 7):
            for y in np.linspace(-0.4, 0.4, 7):
                dist = x**2 + y**2
                opac = max(0.1, 1 - dist*5)
                dots.add(Dot(radius=0.03, color=color, fill_opacity=opac).move_to([x, y, 0]))
        return VGroup(frame, dots)

    def get_image_mockup(color=WHITE):
        frame = Square(side_length=BOX_SIZE, color=color, stroke_width=2, fill_opacity=0.05)
        myocardium = Circle(radius=0.3, color=color, fill_opacity=0.5)
        ventricle = Circle(radius=0.15, color=BLACK, fill_opacity=1)
        return VGroup(frame, myocardium, ventricle)

    # --- 3. GÉNÉRATION DES LIGNES (COILS) ---
    colors = [RED_B, GREEN_B, BLUE_B, PURPLE_B]
    coils_group = VGroup()

    for i, col in enumerate(colors):
        ksp = get_kspace_mockup(col)
        arrow = Arrow(LEFT, RIGHT, color=WHITE, buff=0.1).scale(0.8)
        ifft_lbl = Tex(r"\mathcal{F}_{2D}^{-1}", font_size=28).next_to(arrow, UP, buff=0.1)
        img = get_image_mockup(col)

        row = VGroup(ksp, VGroup(arrow, ifft_lbl), img).arrange(RIGHT, buff=0.8)
        coils_group.add(row)

    # Placement strict dans la moitié basse de l'écran
    coils_group.arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.3)

    # Extraction pour animation individuelle
    kspaces = VGroup(*[row[0] for row in coils_group])
    arrows = VGroup(*[row[1] for row in coils_group])
    images = VGroup(*[row[2] for row in coils_group])

    # Ajout des labels avec les dimensions exactes
    coil_labels = VGroup(*[
        Tex(f"K_{i+1} \\in \\mathbb{{C}}^{{H \\times W}}", color=col, font_size=24).next_to(kspaces[i], LEFT, buff=0.3) 
        for i, col in enumerate(colors)
    ])

    # --- 4. ÉTAPE 1 : IFFT ---
    # L'explication se place juste sous le titre, loin des schémas
    step1_text = Text("2D Inverse Fast Fourier Transform (per coil)", font_size=26, color=BLUE_B)
    step1_text.next_to(title, DOWN, buff=0.3)

    # Formule mathématique rigoureuse
    ifft_formula = Tex(
        r"I_c(x, y) = \iint K_c(k_x, k_y) e^{i 2\pi (k_x x + k_y y)} dk_x dk_y", 
        font_size=36, color=BLUE_B
    ).next_to(step1_text, DOWN, buff=0.2)

    scene.play(Write(step1_text), Write(ifft_formula))
    scene.play(FadeIn(kspaces, RIGHT), Write(coil_labels))
    scene.wait(1)

    # Transformation
    scene.play(FadeIn(arrows, RIGHT))
    scene.play(*[TransformFromCopy(kspaces[i], images[i]) for i in range(4)], run_time=2.5)

    # Précision sur la nature complexe de l'image
    img_labels = VGroup(*[
        Tex(f"I_{i+1} \\in \\mathbb{{C}}^{{H \\times W}}", color=col, font_size=24).next_to(images[i], RIGHT, buff=0.3) 
        for i, col in enumerate(colors)
    ])
    scene.play(Write(img_labels))
    scene.wait(2)

    # --- 5. ÉTAPE 2 : RSS ---
    step2_text = Text("Coil Combination: Root Sum of Squares (Magnitude)", font_size=26, color=YELLOW)
    step2_text.move_to(step1_text.get_center()) # Prend la place exacte du texte précédent

    rss_formula = Tex(
        r"I_{RSS}(x,y) = \sqrt{\sum_{c=1}^{N_c} |I_c(x,y)|^2}", 
        font_size=40, color=YELLOW
    ).next_to(step2_text, DOWN, buff=0.5)

    # Nettoyage du haut de l'écran et des éléments intermédiaires
    scene.play(
        ReplacementTransform(step1_text, step2_text),
        ReplacementTransform(ifft_formula, rss_formula),
        FadeOut(arrows),
        FadeOut(kspaces),
        FadeOut(coil_labels),
        FadeOut(img_labels)
    )
    scene.wait(1)

    # On aligne les images proprement au centre pour la fusion
    scene.play(images.animate.arrange(RIGHT, buff=0.3).move_to(coils_group.get_center()))

    # Création de l'image finale réelle
    final_image = get_image_mockup(WHITE).scale(2.0).move_to(coils_group.get_center())
    final_dim = Tex(r"I_{RSS} \in \mathbb{R}_{+}^{H \times W}", color=WHITE, font_size=28).next_to(final_image, DOWN, buff=0.3)

    # Fusion
    scene.play(*[img.animate.move_to(final_image.get_center()) for img in images], run_time=1.5)
    scene.play(
        ReplacementTransform(images, final_image),
        Write(final_dim),
        run_time=1
    )

    # Effet "Glow"
    glow = final_image[1].copy().set_color(YELLOW).set_opacity(0.8)
    scene.play(FadeIn(glow, scale=1.1), rate_func=there_and_back, run_time=1.5)
    
    scene.wait(3)

    # --- NETTOYAGE FINAL ---
    scene.play(FadeOut(Group(*scene.mobjects)))
