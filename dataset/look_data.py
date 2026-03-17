import napari
import h5py
import numpy as np
from scipy.ndimage import zoom



# --- CONFIGURATION ---
# Remplace par les vrais chemins vers ton TrainingSet
img_path = '../../data/GroundTruth/MultiCoil/Cine/ValidationSet/FullSample/P001/cine_sax.mat'
# Si tu trouves le fichier de label, mets son chemin ici :
label_path = '../../data/GroundTruth/MultiCoil/Cine/ValidationSet/Labels/P001/segmentation.nii.gz' 

def load_reconstructed_mat(path):
    with h5py.File(path, 'r') as f:
        # Dans Cine, il y a souvent une dimension Temps (t)
        # Forme probable : (bobines, temps, coupes, y, x)
        raw = f['kspace_full'] # Vérifie le nom de la clé !
        kspace = raw['real'] + 1j * raw['imag']

        # Reconstruction iFFT
        img_complex = np.fft.ifftshift(np.fft.ifft2(kspace, axes=(-2, -1)), axes=(-2, -1))
        # Magnitude + SOS
        volume = np.sqrt(np.sum(np.abs(img_complex)**2, axis=0))
        
        # Si c'est du Cine (4D), on prend la première frame temporelle pour la 3D
        if volume.ndim == 4:
            volume = volume[0] 
        return volume



# 1. Charger l'image
volume_img = load_reconstructed_mat(img_path)

# 2. Zoom pour rendre le volume "carré" (Indispensable pour Cine SAX)
# Les coupes SAX sont très espacées, on étire l'axe Z
volume_resampled = zoom(volume_img, (5.0, 1, 1), order=1)

# 3. Lancer Napari
viewer = napari.Viewer()

# Ajouter l'image
viewer.add_image(volume_resampled, name='IRM_Cine_P001', colormap='gray')

# 4. Ajouter le Label (si tu l'as trouvé)
try:
    import nibabel as nib
    label_data = nib.load(label_path).get_fdata()
    # On applique le même zoom que l'image pour qu'ils se superposent
    label_resampled = zoom(label_data, (5.0, 1, 1), order=0) # order 0 pour les labels !
    viewer.add_labels(label_resampled.astype(int), name='Segmentation')
except:
    print("Label non trouvé, affichage de l'image seule.")

viewer.dims.ndisplay = 3
napari.run()
