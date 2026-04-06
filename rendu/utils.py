from manimlib import *



def cleanup(scene):
    # Nettoyage rigoureux de tous les objets pour la scène suivante
    scene.play(FadeOut(Group(*scene.mobjects)), run_time=1)
