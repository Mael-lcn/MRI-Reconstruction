import numpy as np
import scipy.io as sio
import h5py
import scipy.fft as sp_fft



def loadmat(file_path):
    """
    Lit un fichier .mat de manière robuste.
    Gère les anciens formats MATLAB (via scipy) et les nouveaux v7.3 (via h5py).
    """
    try:
        mat_data = {}
        with h5py.File(file_path, 'r') as f:
            for k, v in f.items():
                # On conserve l'ordre natif de h5py qui aligne les dimensions
                # spatiales à la fin : [frames, slices, coils, y, x]
                mat_data[k] = np.array(v)
        return mat_data
    except Exception as e:
        raise RuntimeError(f"Impossible de lire le fichier {file_path} : {str(e)}")


def multicoilkdata2img(kdata):
    """
    Transforme l'espace K (k-space) multi-antennes en images spatiales réelles.
    Utilise scipy.fft pour bénéficier de l'accélération multicoeur (PocketFFT).
    """
    # 1. IFFT 2D sur les deux derniers axes (Hauteur, Largeur)
    img_complex = sp_fft.ifftshift(
        sp_fft.ifft2(
            sp_fft.ifftshift(kdata, axes=(-2, -1)), 
            axes=(-2, -1)
        ), 
        axes=(-2, -1)
    )

    # 2. Combinaison des bobines (Coils) par "Root Sum of Squares" (RSS)
    # Dans CMRxRecon, l'axe des bobines est l'antépénultième (index -3)
    img_rss = np.sqrt(np.sum(np.abs(img_complex)**2, axis=-3))
    
    return img_rss
