import os
import time
import argparse
import multiprocessing
import traceback
import h5py
import gc
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from loadFun import load_h5_slice, multicoilkdata2img_slice



def paddingZero_np(np_data: np.ndarray, target_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Applique un remplissage de zéros (padding) ou un recadrage central (cropping) dynamique sur un tenseur.
    
    Alloue directement la matrice cible et y injecte les données afin d'optimiser 
    l'utilisation de la mémoire vive, en évitant la duplication des objets.

    Args:
        np_data (np.ndarray): Tenseur source contenant les données complexes ou réelles.
        target_shape (Tuple[int, int]): Dimensions spatiales cibles (Hauteur, Largeur).

    Returns:
        np.ndarray: Tenseur redimensionné aux dimensions cibles.
    """
    shape = list(np_data.shape)
    H, W = shape[-2], shape[-1]

    if H == target_shape[0] and W == target_shape[1]:
        return np_data

    shape[-2], shape[-1] = target_shape[0], target_shape[1]
    padded_data = np.zeros(shape, dtype=np_data.dtype)

    src_h_start = max((H - target_shape[0]) // 2, 0)
    src_w_start = max((W - target_shape[1]) // 2, 0)
    src_h_end = src_h_start + min(H, target_shape[0])
    src_w_end = src_w_start + min(W, target_shape[1])

    dst_h_start = max((target_shape[0] - H) // 2, 0)
    dst_w_start = max((target_shape[1] - W) // 2, 0)
    dst_h_end = dst_h_start + min(H, target_shape[0])
    dst_w_end = dst_w_start + min(W, target_shape[1])

    padded_data[..., dst_h_start:dst_h_end, dst_w_start:dst_w_end] = \
        np_data[..., src_h_start:src_h_end, src_w_start:src_w_end]

    return padded_data


def process_single_slice(args: Tuple) -> Dict[str, Any]:
    """
    Exécute le pipeline de traitement complet pour une unique tranche spatiale d'IRM.
    
    Le traitement inclut le chargement paresseux, l'application du masque de sous-échantillonnage,
    la conversion vers l'espace image (IFFT), et la normalisation. L'empreinte mémoire est 
    strictement contrôlée via des suppressions manuelles (del).

    Args:
        args (Tuple): Contient les paramètres de la tranche à traiter et les chemins 
                      de destination cloisonnés.

    Returns:
        Dict[str, Any]: Dictionnaire de résultats contenant le statut de l'opération, 
                        les identifiants de la tranche, et le nombre de trames temporelles extraites.
    """
    full_path, mask_path, slice_idx, axis_name, item, save_full_dir, save_04_dir, coilInfo = args
    result = {
        "status": "SUCCESS", 
        "item": item, 
        "slice_idx": slice_idx, 
        "axis": axis_name, 
        "num_frames": 0, 
        "msg": ""
    }

    try:
        # 1. Chargement paresseux des données
        raw_full = load_h5_slice(full_path, slice_idx, dataset_name='kspace_full')
        raw_mask = load_h5_slice(mask_path, slice_idx)

        # 2. Conversion vers l'espace complexe avec allocation directe
        data_full = np.empty(raw_full.shape, dtype=np.complex64)
        if raw_full.dtype.names is not None and 'real' in raw_full.dtype.names:
            data_full.real, data_full.imag = raw_full['real'], raw_full['imag']
        else:
            data_full[:] = raw_full
        del raw_full 
        
        mask_04 = np.empty(raw_mask.shape, dtype=np.complex64)
        if raw_mask.dtype.names is not None and 'real' in raw_mask.dtype.names:
            mask_04.real, mask_04.imag = raw_mask['real'], raw_mask['imag']
        else:
            mask_04[:] = raw_mask
        del raw_mask

        # 3. Application du masque et ajustement des dimensions (Padding)
        data_04 = data_full * mask_04
        del mask_04
        
        data_full_padded = paddingZero_np(data_full, (512, 512))
        del data_full
        data_04_padded = paddingZero_np(data_04, (512, 512))
        del data_04

        # 4. Reconstruction spatiale (Transformée de Fourier inverse)
        imgs_full = multicoilkdata2img_slice(data_full_padded)
        del data_full_padded
        imgs_04 = multicoilkdata2img_slice(data_04_padded)
        del data_04_padded

        # Extraction de la dimension temporelle pour les statistiques du jeu de données
        result["num_frames"] = imgs_full.shape[0]

        # 5. Normalisation par le maximum d'intensité
        max_f = np.amax(imgs_full, axis=(1, 2), keepdims=True)
        np.divide(imgs_full, max_f, out=imgs_full, where=max_f!=0)
        del max_f

        max_4 = np.amax(imgs_04, axis=(1, 2), keepdims=True)
        np.divide(imgs_04, max_4, out=imgs_04, where=max_4!=0)
        del max_4

        # 6. Création des répertoires cibles (JIT) et sauvegarde
        os.makedirs(save_full_dir, exist_ok=True)
        os.makedirs(save_04_dir, exist_ok=True)

        file_name = f"{item}_{coilInfo}_{axis_name}_slice{slice_idx:02d}.npy"
        np.save(os.path.join(save_full_dir, file_name), imgs_full)
        np.save(os.path.join(save_04_dir, file_name), imgs_04)
        
        # Libération finale de la mémoire allouée au processus
        del imgs_full, imgs_04
        gc.collect()
        
        return result

    except Exception as e:
        result["status"] = "ERROR"
        result["msg"] = f"Échec sur le patient {item} (axe {axis_name}, tranche {slice_idx}):\n{traceback.format_exc()}"
        return result


def generate_slice_tasks(patient_dir: str, item: str, save_dir: str, coilInfo: str) -> List[Tuple]:
    """
    Analyse les données d'un patient pour générer la liste des tâches de traitement.
    
    Intègre une logique de vérification de l'existant (skip logic) pour éviter
    le retraitement inutile de données déjà consolidées sur le disque.

    Args:
        patient_dir (str): Chemin vers le répertoire source du patient.
        item (str): Identifiant du patient (ex: 'P001').
        save_dir (str): Répertoire de destination cloisonné (ex: .../TrainingSet).
        coilInfo (str): Configuration des antennes.

    Returns:
        List[Tuple]: Liste des arguments prêts à être distribués aux processus travailleurs.
    """
    tasks = []
    
    save_full_dir = os.path.join(save_dir, "FullSample", item)
    save_04_dir = os.path.join(save_dir, "AccFactor04", item)

    # Vérification de l'existence de données préalablement calculées
    if os.path.exists(save_full_dir) and os.path.exists(save_04_dir):
        full_files = [f for f in os.listdir(save_full_dir) if f.endswith('.npy')]
        acc_files = [f for f in os.listdir(save_04_dir) if f.endswith('.npy')]
        
        if len(full_files) > 0 and len(acc_files) > 0:
            print(f"    [Information] {item} ignoré : Patient déjà traité ({len(full_files)} fichiers existants).")
            return [] 

    mask_dir = patient_dir.replace('FullSample', 'Mask_Task1')
    axes = [
        ("lax", os.path.join(patient_dir, "cine_lax.mat"), os.path.join(mask_dir, "cine_lax_mask_Uniform4.mat")),
        ("sax", os.path.join(patient_dir, "cine_sax.mat"), os.path.join(mask_dir, "cine_sax_mask_Uniform4.mat"))
    ]

    for axis_name, full_path, mask_path in axes:
        if os.path.exists(full_path) and os.path.exists(mask_path):
            try:
                with h5py.File(full_path, 'r') as f:
                    num_slices = f['kspace_full'].shape[1] 
                
                for slice_idx in range(num_slices):
                    tasks.append((full_path, mask_path, slice_idx, axis_name, item, save_full_dir, save_04_dir, coilInfo))
            except Exception as e:
                print(f"    [Avertissement] {item} : Échec de la lecture de l'axe {axis_name} -> {e}")
                
    if len(tasks) > 0:
        print(f"    [Planification] {item} : {len(tasks)} tranches ajoutées à la file d'attente.")
        
    return tasks


def generate_pairs(dataset_out_dir: str, acc_factor: str = "AccFactor04") -> Tuple[int, str]:
    """ 
    Génère le fichier texte de mapping reliant les images sous-échantillonnées à leur vérité terrain.
    
    Parcourt récursivement l'arborescence cloisonnée (TrainingSet, ValidationSet, ou TestSet) 
    pour lier chaque fichier .npy. Ce fichier est destiné à être ingéré par l'objet Dataset de PyTorch.

    Args:
        dataset_out_dir (str): Répertoire cloisonné cible (ex: output/TrainingSet).
        acc_factor (str, optional): Nom du dossier contenant les entrées du modèle.

    Returns:
        Tuple[int, str]: Nombre total de paires générées et chemin vers le fichier de mapping.
    """
    imgs_dir_base = os.path.join(dataset_out_dir, acc_factor)
    full_sample_dir_base = os.path.join(dataset_out_dir, 'FullSample')
    pairs_count = 0

    if not os.path.exists(imgs_dir_base) or not os.path.exists(full_sample_dir_base):
        return 0, "Dossiers cibles introuvables."

    # Sauvegarde du fichier à la racine du sous-dossier de set (ex: output/TrainingSet/AccFactor04_pairs.txt)
    file_path = os.path.join(dataset_out_dir, f"{acc_factor}_pairs.txt")

    try:
        with open(file_path, "w") as file_obj:
            for patient_folder in sorted(os.listdir(imgs_dir_base)):
                patient_imgs_dir = os.path.join(imgs_dir_base, patient_folder)
                patient_gt_dir = os.path.join(full_sample_dir_base, patient_folder)

                if not os.path.isdir(patient_imgs_dir):
                    continue 

                for img_name in sorted(os.listdir(patient_imgs_dir)):
                    if img_name.endswith('.npy'):
                        img_path = os.path.join(patient_imgs_dir, img_name)
                        gt_path = os.path.join(patient_gt_dir, img_name)
                        
                        if os.path.exists(gt_path):
                            file_obj.write(f"{img_path} {gt_path}\n")
                            pairs_count += 1
                            
        return pairs_count, file_path
    except Exception as e:
        return 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Pipeline de prétraitement HPC pour la reconstruction IRM (CMRxRecon)")
    parser.add_argument('-i', "--input", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data_pre/home2/Raw_data/MICCAIChallenge2024/ChallengeData", help="Répertoire source principal (Training/Validation)")
    parser.add_argument('-t', "--test_input", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data_post/GroundTruth", help="Répertoire source spécifique au jeu de test")
    parser.add_argument('-o', "--output", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data", help="Répertoire de destination consolidé")
    parser.add_argument('-w', "--workers", type=int, default=max(1, multiprocessing.cpu_count()-1), help="Nombre de processus de calcul alloués")

    args = parser.parse_args()
    coilInfo = 'MultiCoil'
    
    # Cartographie des partitions vers leurs répertoires sources respectifs
    dataset_splits = [
        ('TrainingSet', args.input),
        ('ValidationSet', args.input),
        ('TestSet', args.test_input)
    ]
    
    all_slice_tasks = []

    print("---------------------------------------------------------")
    print(" Phase 1 : Planification des tâches et analyse du disque")
    print("---------------------------------------------------------")
    
    for split_name, base_input_dir in dataset_splits:
        dir_path = os.path.join(base_input_dir, coilInfo, 'Cine', split_name, 'FullSample')
        save_dir = os.path.join(args.output, split_name)

        if not os.path.isdir(dir_path):
            print(f"\n  [Information] Séquence '{split_name}' introuvable à la source, ignorée.")
            continue
            
        print(f"\n  Analyse de la séquence : {split_name}...")
        for item in sorted(os.listdir(dir_path)):
            patient_dir = os.path.join(dir_path, item)
            if os.path.isdir(patient_dir):
                all_slice_tasks.extend(generate_slice_tasks(patient_dir, item, save_dir, coilInfo))

    total_tasks = len(all_slice_tasks)
    if total_tasks == 0:
        print("\n[Information] L'intégralité du jeu de données a déjà été traitée. Fin de l'exécution.")
        return

    print(f"\n=> Total à calculer : {total_tasks} nouvelles tranches réparties sur {args.workers} processus.")
    
    start_time = time.time()
    success_count = 0
    errors = 0
    total_2d_frames_extracted = 0

    print("\n---------------------------------------------------------")
    print(" Phase 2 : Exécution distribuée de la reconstruction")
    print("---------------------------------------------------------")
    with multiprocessing.Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_slice, all_slice_tasks), total=total_tasks, desc="Progression globale"):
            if result["status"] == "SUCCESS":
                success_count += 1
                total_2d_frames_extracted += result.get("num_frames", 0)
            else:
                errors += 1
                tqdm.write(result["msg"])

    print("\n---------------------------------------------------------")
    print(" Phase 3 : Consolidation des descripteurs pour l'apprentissage")
    print("---------------------------------------------------------")
    
    total_pairs_generated = 0
    for split_name, _ in dataset_splits:
        split_out_dir = os.path.join(args.output, split_name)
        if os.path.isdir(split_out_dir):
            p_count, p_path = generate_pairs(split_out_dir)
            if p_count > 0:
                print(f"  [{split_name}] Fichier de registre généré : {p_path} ({p_count} séquences)")
                total_pairs_generated += p_count

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    report = f"""
        =========================================================
                    RAPPORT D'EXÉCUTION DU PIPELINE              
        =========================================================
        Durée totale de l'opération   : {int(hours)}h {int(minutes)}m {int(seconds)}s
        Tranches spatiales calculées  : {success_count} / {total_tasks}
        ---------------------------------------------------------
        Métriques Deep Learning :
        Fichiers tenseurs produits  : {total_pairs_generated * 2} (Entrée + Cible)
        Images 2D effectives        : ~{total_2d_frames_extracted} (Frames nouvellement extraites)
        Total Paires consolidées    : {total_pairs_generated}
        ---------------------------------------------------------
        Erreurs d'exécution système   : {errors}
        =========================================================
        """
    print(report)


if __name__ == "__main__":
    main()
