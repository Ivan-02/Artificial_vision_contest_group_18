import os
import zipfile
import shutil
import pandas as pd
import cv2
import yaml
from tqdm import tqdm
import json
from .common.data_utils import GameInfoParser, GeometryUtils


class DataManager:

    def __init__(self, config):
        self.cfg = config
        self.raw_data_path = self.cfg['paths']['raw_data']
        self.yolo_dataset_path = self.cfg['paths']['yolo_dataset']

    def prepare_dataset(self):
        self._unzip_and_delete(self.raw_data_path, self.raw_data_path)

        # --- NOVITÀ: Rinomina cartelle e copia ROI PRIMA della generazione del dataset ---

        # 1. Rinomina le cartelle NSMOT-XX -> XX
        # È fondamentale farlo PRIMA della conversione YOLO affinché i nomi dei file
        # nel dataset finale siano puliti (es. "01_frame_00.txt" invece di "NSMOT-01_frame...")
        self._rename_video_folders()

        # 2. Distribuzione del file roi.json nelle sottocartelle
        self._distribute_roi_json()

        # --------------------------------------------------------------------------------

        subsets = ['train', 'test']
        found_any = False

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if os.path.exists(subset_path):
                print(f"Trovato subset: {subset}. Inizio conversione...")
                found_any = True
                self._convert_mot_to_yolo(
                    source_dir=subset_path,
                    output_dir=self.yolo_dataset_path,
                    sub_folder=subset
                )

        if not found_any:
            print("Nessuna sottocartella standard trovata. Tento conversione nella root...")
            self._convert_mot_to_yolo(self.raw_data_path, self.yolo_dataset_path, sub_folder='train')

        self._create_yolo_yaml()

        print("---Preparazione behavior GT")
        self.prepare_behavior_gt()

        print("---Rimozione classe ball dalla GT")
        self.remove_ball_from_all_gt()

        print("--- Preparazione Dataset Completata ---\n")

    def rename_video_folders(self):
        """
        Scansiona le cartelle train/test e rinomina le cartelle video
        da 'SNMOT-XX' a 'XX'.
        """
        print("--- Standardizzazione nomi cartelle video (SNMOT-XX -> XX) ---")
        subsets = ['train', 'test']

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if not os.path.exists(subset_path):
                continue

            # Itera sulle cartelle
            for folder_name in os.listdir(subset_path):
                folder_path = os.path.join(subset_path, folder_name)

                if os.path.isdir(folder_path) and folder_name.startswith("SNMOT-"):
                    # Estrae l'indice (es. SNMOT-01 -> 01)
                    try:
                        new_name = folder_name.split('-')[1]
                        new_path = os.path.join(subset_path, new_name)

                        # Evita sovrascritture se la cartella esiste già
                        if not os.path.exists(new_path):
                            os.rename(folder_path, new_path)
                            print(f"Rinominato: {folder_name} -> {new_name}")
                        else:
                            print(f"Warning: Impossibile rinominare {folder_name}, {new_name} esiste già.")
                    except IndexError:
                        print(f"Skipping format non atteso: {folder_name}")

    def _distribute_roi_json(self):
        """
        Copia il file roi.json:
        1. In ogni sottocartella (train/test).
        2. All'interno di ogni cartella video in train/test.
        """
        print("--- Distribuzione file ROI.json ---")
        source_roi = os.path.join('./configs', 'roi.json')  # Path relativo come richiesto
        # Fallback se non lo trova nel path relativo, prova a usare config se definito
        if not os.path.exists(source_roi) and 'roi' in self.cfg['paths']:
            source_roi = self.cfg['paths']['roi']

        if not os.path.exists(source_roi):
            print(f"Errore: File ROI sorgente non trovato in {source_roi}")
            return

        subsets = ['train', 'test']

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if not os.path.exists(subset_path):
                continue

            # 1. Copia nella root del subset (es. raw_data/train/roi.json)
            dst_subset = os.path.join(subset_path, 'roi.json')
            shutil.copy(source_roi, dst_subset)

            # 2. Copia in ogni cartella video
            video_folders = [f for f in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, f))]
            for video_name in video_folders:
                video_path = os.path.join(subset_path, video_name)
                dst_video = os.path.join(video_path, 'roi.json')
                shutil.copy(source_roi, dst_video)

        print("Distribuzione ROI completata.")

    def prepare_behavior_gt(self):
        """
        Genera i file di Ground Truth per la behavior analysis.
        Input: ROI json (self.cfg['paths']['roi'])
        Output: behavior_XX_gt.txt nella cartella gt di ogni video.
        """
        print("--- Inizio generazione GT Behavior Analysis ---")

        roi_path = self.cfg['paths']['roi']
        if not os.path.exists(roi_path):
            print(f"Errore: File ROI non trovato in {roi_path}")
            return

        with open(roi_path, 'r') as f:
            rois = json.load(f)

        subsets = ['train', 'test']

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if not os.path.exists(subset_path):
                continue

            video_folders = [f for f in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, f))]

            for video_name in tqdm(video_folders, desc=f"Generazione GT Behavior {subset}"):
                video_path = os.path.join(subset_path, video_name)
                gt_folder = os.path.join(video_path, 'gt')
                gt_txt_path = os.path.join(gt_folder, 'gt.txt')
                ini_path = os.path.join(video_path, 'gameinfo.ini')
                img_dir = os.path.join(video_path, 'img1')

                output_filename = f"behavior_gt.txt"
                output_path = os.path.join(gt_folder, output_filename)

                if os.path.exists(output_path):
                    continue

                if not os.path.exists(gt_txt_path):
                    continue

                id_map = GameInfoParser.get_id_map(ini_path)
                ball_ids = [k for k, v in id_map.items() if "ball" in v]

                try:
                    df = pd.read_csv(gt_txt_path, header=None)
                    df.columns = ['frame_id', 'obj_id', 'x', 'y', 'w', 'h', 'conf', 'cls', 'vis', 'u']
                except Exception as e:
                    print(f"Errore lettura GT per {video_name}: {e}")
                    continue

                # Recupero dimensioni immagine per normalizzazione/verifica ROI
                h_img, w_img = 1080, 1920
                if os.path.exists(img_dir):
                    images = os.listdir(img_dir)
                    if images:
                        img = cv2.imread(os.path.join(img_dir, images[0]))
                        if img is not None:
                            h_img, w_img = img.shape[:2]

                frames = sorted(df['frame_id'].unique())

                with open(output_path, 'w') as f_out:
                    for frame_id in frames:
                        frame_data = df[df['frame_id'] == frame_id]
                        counts = {1: 0, 2: 0}

                        for _, row in frame_data.iterrows():
                            obj_id = int(row['obj_id'])

                            if obj_id in ball_ids:
                                continue

                            # 2. Conversione Coordinate per GeometryUtils
                            # GT fornisce Top-Left (x, y), GeometryUtils si aspetta Center (cx, cy)
                            # per calcolare correttamente i "piedi" (base_y).
                            cx = row['x'] + (row['w'] / 2.0)
                            cy = row['y'] + (row['h'] / 2.0)

                            # Creiamo il box formato [cx, cy, w, h]
                            box_xywh = [cx, cy, row['w'], row['h']]

                            # 3. Verifica inclusione usando GeometryUtils
                            if GeometryUtils.is_in_roi(box_xywh, rois.get('roi1'), w_img, h_img):
                                counts[1] += 1
                            elif GeometryUtils.is_in_roi(box_xywh, rois.get('roi2'), w_img, h_img):
                                counts[2] += 1

                        f_out.write(f"{int(frame_id)},1,{counts[1]}\n")
                        f_out.write(f"{int(frame_id)},2,{counts[2]}\n")
                        f_out.flush()

        print("--- Generazione GT Behavior Completata ---\n")

    def _create_yolo_yaml(self):
        """Crea il file dataset.yaml per YOLO."""
        yaml_path = os.path.join(self.yolo_dataset_path, 'dataset.yaml')
        abs_path = os.path.abspath(self.yolo_dataset_path)

        data = {
            'path': abs_path,
            'train': 'images/train',
            'val': 'images/train',  # Default fallback
            'test': 'images/test',
            'nc': 4,
            'names': {0: 'player_uf', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        }

        # Gestione path validazione intelligente
        if not os.path.exists(os.path.join(abs_path, 'images', 'test')):
            if 'test' in data: del data['test']

        if os.path.exists(os.path.join(abs_path, 'images', 'valid')):
            data['val'] = 'images/valid'
        elif os.path.exists(os.path.join(abs_path, 'images', 'test')):
            data['val'] = 'images/test'

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"File di configurazione YOLO creato: {yaml_path}")

    @staticmethod
    def _unzip_and_delete(source_folder, output_folder):
        if not os.path.exists(source_folder): return
        if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)

        for item in os.listdir(source_folder):
            if item.endswith(".zip"):
                zip_path = os.path.join(source_folder, item)
                print(f"Estraendo: {item}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(output_folder)
                    os.remove(zip_path)
                except Exception as e:
                    print(f"Errore zip {item}: {e}")

    def _convert_mot_to_yolo(self, source_dir, output_dir, sub_folder='train'):
        images_out = os.path.join(output_dir, 'images', sub_folder)
        labels_out = os.path.join(output_dir, 'labels', sub_folder)
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        video_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

        for video_name in tqdm(video_folders, desc=f"Processando {sub_folder}"):
            video_path = os.path.join(source_dir, video_name)

            # Utilizzo Parser condiviso
            ini_path = os.path.join(video_path, 'gameinfo.ini')
            id_to_label = GameInfoParser.get_id_map(ini_path)

            gt_path = os.path.join(video_path, 'gt', 'gt.txt')
            img_dir = os.path.join(video_path, 'img1')

            if not os.path.exists(gt_path) or not os.path.exists(img_dir):
                continue

            try:
                df = pd.read_csv(gt_path, header=None)
                df.columns = ['frame_id', 'obj_id', 'x', 'y', 'w', 'h', 'conf', 'cls', 'vis', 'u']
            except:
                continue

            grouped = df.groupby('frame_id')

            for frame_id, group in grouped:
                unique_name = f"{video_name}_frame_{int(frame_id):06d}"
                label_file = os.path.join(labels_out, unique_name + '.txt')

                if os.path.exists(label_file): continue

                # Gestione estensioni immagine
                img_name_base = f"{int(frame_id):06d}"
                src_img = os.path.join(img_dir, img_name_base + ".jpg")
                if not os.path.exists(src_img):
                    src_img = os.path.join(img_dir, img_name_base + ".png")
                if not os.path.exists(src_img): continue

                # Lettura dimensione immagine necessaria per normalizzazione
                img = cv2.imread(src_img)
                if img is None: continue
                h_img, w_img = img.shape[:2]

                yolo_lines = []
                for _, row in group.iterrows():
                    obj_id = int(row['obj_id'])
                    label_str = id_to_label.get(obj_id, "unknown")

                    # Logica Mapping Classi
                    final_class_id = -1
                    if "ball" in label_str:
                        final_class_id = 0
                    elif "goalkeeper" in label_str:
                        final_class_id = 1
                    elif "player" in label_str:
                        final_class_id = 2
                    elif "referee" in label_str:
                        final_class_id = 3

                    if final_class_id == -1: continue

                    # Conversione coordinate: TopLeft -> Center + Normalizzazione
                    x_c = (row['x'] + row['w'] / 2) / w_img
                    y_c = (row['y'] + row['h'] / 2) / h_img
                    w_n = row['w'] / w_img
                    h_n = row['h'] / h_img

                    # Clamping [0, 1]
                    x_c = max(0, min(1, x_c))
                    y_c = max(0, min(1, y_c))
                    w_n = max(0, min(1, w_n))
                    h_n = max(0, min(1, h_n))

                    yolo_lines.append(f"{final_class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

                if yolo_lines:
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(yolo_lines))

                    # Symlink o copia immagine
                    ext = os.path.splitext(src_img)[1]
                    dst_img = os.path.join(images_out, unique_name + ext)

                    if os.path.exists(dst_img) or os.path.islink(dst_img):
                        try:
                            os.remove(dst_img)
                        except:
                            pass
                    try:
                        os.symlink(os.path.abspath(src_img), os.path.abspath(dst_img))
                    except:
                        shutil.copy(os.path.abspath(src_img), os.path.abspath(dst_img))

    def remove_ball_from_all_gt(self):
        """
        Versione OTTIMIZZATA:
        1. Itera su tutti i video.
        2. Se trova 'gt.txt.bak', SALTA il video (già processato).
        3. Se non lo trova, rimuove la palla e crea il backup.
        """
        print("--- AVVIO RIMOZIONE PALLA (con skip dei già processati) ---")

        subsets = ['train', 'test']
        total_modified = 0
        total_skipped = 0
        total_errors = 0

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if not os.path.exists(subset_path):
                continue

            video_folders = [f for f in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, f))]

            for video_name in tqdm(video_folders, desc=f"Elaborazione {subset}"):
                video_path = os.path.join(subset_path, video_name)
                gt_path = os.path.join(video_path, 'gt', 'gt.txt')
                backup_path = gt_path + ".bak"  # Percorso del backup
                ini_path = os.path.join(video_path, 'gameinfo.ini')

                # 1. CONTROLLO SKIP: Se esiste il backup, il video è già fatto.
                if os.path.exists(backup_path):
                    total_skipped += 1
                    continue

                # Controllo esistenza file sorgenti
                if not os.path.exists(gt_path) or not os.path.exists(ini_path):
                    continue

                try:
                    # 2. Logica di rimozione (eseguita solo se non c'è backup)
                    id_map = GameInfoParser.get_id_map(ini_path)
                    ball_ids = [k for k, v in id_map.items() if "ball" in str(v).lower()]

                    if not ball_ids:
                        continue

                    df = pd.read_csv(gt_path, header=None)
                    if df.empty: continue

                    df.columns = ['frame_id', 'obj_id', 'x', 'y', 'w', 'h', 'conf', 'cls', 'vis', 'u']

                    original_count = len(df)
                    df_clean = df[~df['obj_id'].isin(ball_ids)]
                    new_count = len(df_clean)

                    # 3. Salvataggio e creazione backup
                    if new_count < original_count:
                        shutil.copy(gt_path, backup_path)  # Crea il backup ORA
                        df_clean.to_csv(gt_path, header=False, index=False)
                        total_modified += 1
                    else:
                        pass

                except Exception as e:
                    print(f"Errore su {video_name}: {e}")
                    total_errors += 1

        print("\n--- OPERAZIONE COMPLETATA ---")
        print(f"Video modificati ora: {total_modified}")
        print(f"Video saltati (già fatti): {total_skipped}")
        print(f"Errori: {total_errors}")