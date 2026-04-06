from manimlib import *



def play_diffcmr_architecture_scene(scene):
    """
    """
    title = Text("DiffCMR: Conditional Diffusion Architecture", color=GOLD).to_edge(UP, buff=0.2)
    subtitle = Text("Part 1: Overall Pipeline", font_size=28, color=LIGHT_BROWN).next_to(title, DOWN, buff=0.2)

    # Animation des textes
    scene.play(FadeIn(title, shift=DOWN))
    scene.wait(0.2)
    scene.play(Write(subtitle))
    scene.wait(0.5)

    # === 2. PRÉPARATION DES IMAGES ===
    archi_img = ImageMobject("images/archi.png")
    unet_img = ImageMobject("images/3D-U-Net-network-architecture.png")

    # === 3. SÉQUENCE ARCHI.PNG (FOCUS GAUCHE -> DROITE -> ENTIER) ===
    # Initialisation zoomée sur la gauche (entrées)
    archi_img.set_height(6.5) 
    archi_img.move_to(RIGHT * 3.5 + DOWN * 1) 

    scene.play(FadeIn(archi_img))
    scene.wait(1)

    # Glissement panoramique vers la droite
    scene.play(
        archi_img.animate.move_to(LEFT * 3.5 + DOWN * 1),
        run_time=3,
        rate_func=linear
    )
    scene.wait(1)

    # De-zoom global
    scene.play(
        archi_img.animate.set_height(4.5).move_to(DOWN * 1.2), # On descend un peu l'image globale
        run_time=1.5
    )
    scene.wait(1)

    # === 4. HIGHLIGHT DU NOEUD SOMMATION (+) ===
    sum_highlight = Circle(radius=0.35, color=YELLOW, stroke_width=6)
    # Coordonnées ajustées pour viser le (+)
    sum_highlight.move_to(archi_img.get_center() + LEFT * 1.5 + DOWN * 1.1)

    scene.play(ShowCreation(sum_highlight))
    scene.play(Flash(sum_highlight, color=YELLOW, flash_radius=0.5))
    scene.wait(1)
    scene.play(FadeOut(sum_highlight))

    # === 5. TRANSITION VERS LE BLOC E ===
    new_subtitle = Text("Part 2: Detailed U-Net Architecture", font_size=28, color=LIGHT_BROWN).next_to(title, DOWN, buff=0.2)
    
    # Point de zoom sur le bloc bleu E
    target_point = archi_img.get_center() + RIGHT * 1.1 + DOWN * 1.1

    scene.play(
        Transform(subtitle, new_subtitle),
        archi_img.animate.scale(8, about_point=target_point).fade(1),
        run_time=1.5
    )

    # === 6. SÉQUENCE 3D-U-NET ===
    unet_img.set_width(11) # Légère réduction pour l'équilibre visuel
    # On positionne l'image SOUS le sous-titre pour éviter tout chevauchement
    unet_img.next_to(new_subtitle, DOWN, buff=0.5)

    scene.play(FadeIn(unet_img, scale=0.5), run_time=1.5)
    
    # Effet de respiration lente pour la fin
    scene.play(
        unet_img.animate.scale(1.05),
        run_time=5,
        rate_func=linear
    )
    scene.wait(1)

    # Sortie finale
    scene.play(
        FadeOut(unet_img),
        FadeOut(title),
        FadeOut(subtitle),
        run_time=1
    )
