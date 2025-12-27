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
        print(f"\n[DataManager] {'=' * 50}")
        print("[DataManager] AVVIO: PREPARAZIONE DATASET & FORMATTING")
        print(f"[DataManager] {'=' * 50}")
        print(f"[DataManager] Raw Data Path : {self.raw_data_path}")
        print(f"[DataManager] YOLO Output   : {self.yolo_dataset_path}")

        print("\n[DataManager] [1/7] Estrazione archivi ZIP...")
        self._unzip_and_delete(self.raw_data_path, self.raw_data_path)

        print("\n[DataManager] [2/7] Standardizzazione nomi cartelle...")
        self._rename_video_folders()

        print("\n[DataManager] [3/7] Distribuzione configurazioni ROI...")
        self._distribute_roi_json()

        print("\n[DataManager] [4/7] Conversione formato MOT -> YOLO...")
        subsets = ['train', 'test']
        found_any = False

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if os.path.exists(subset_path):
                print(f"[DataManager]       Trovato subset: '{subset}'")
                found_any = True
                self._convert_mot_to_yolo(
                    source_dir=subset_path,
                    output_dir=self.yolo_dataset_path,
                    sub_folder=subset
                )

        if not found_any:
            print("[DataManager] [!] Nessuna sottocartella standard (train/test) trovata.")
            print("[DataManager]     Tento conversione nella root come 'train'...")
            self._convert_mot_to_yolo(self.raw_data_path, self.yolo_dataset_path, sub_folder='train')

        print("\n[DataManager] [5/7] Generazione dataset.yaml...")
        self._create_yolo_yaml()

        print("\n[DataManager] [6/7] Generazione Ground Truth per Behavior Analysis...")
        self.prepare_behavior_gt()

        print("\n[DataManager] [7/7] Pulizia classi (Rimozione 'Ball' dalla GT)...")
        self.remove_ball_from_all_gt()

        print(f"\n[DataManager] {'=' * 50}")
        print("[DataManager] [✔] PREPARAZIONE DATASET COMPLETATA")
        print(f"[DataManager] {'=' * 50}\n")

    def rename_video_folders(self):
        subsets = ['train', 'test']
        count = 0

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if not os.path.exists(subset_path):
                continue

            for folder_name in os.listdir(subset_path):
                folder_path = os.path.join(subset_path, folder_name)

                if os.path.isdir(folder_path) and folder_name.startswith("SNMOT-"):
                    try:
                        new_name = folder_name.split('-')[1]
                        new_path = os.path.join(subset_path, new_name)

                        if not os.path.exists(new_path):
                            os.rename(folder_path, new_path)
                            count += 1
                        else:
                            print(f"[DataManager] [WARN] Impossibile rinominare {folder_name}, {new_name} esiste già.")
                    except IndexError:
                        print(f"[DataManager] [WARN] Skipping format non atteso: {folder_name}")

        if count > 0:
            print(f"[DataManager]       Rinominati {count} video.")
        else:
            print(f"[DataManager]       Nessuna cartella da rinominare trovata.")

    def _distribute_roi_json(self):
        source_roi = os.path.join('./configs', 'roi.json')
        if not os.path.exists(source_roi) and 'roi' in self.cfg['paths']:
            source_roi = self.cfg['paths']['roi']

        if not os.path.exists(source_roi):
            print(f"[DataManager] [ERROR] File ROI sorgente non trovato in {source_roi}")
            return

        subsets = ['train', 'test']
        count = 0

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if not os.path.exists(subset_path):
                continue

            dst_subset = os.path.join(subset_path, 'roi.json')
            shutil.copy(source_roi, dst_subset)

            video_folders = [f for f in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, f))]
            for video_name in video_folders:
                video_path = os.path.join(subset_path, video_name)
                dst_video = os.path.join(video_path, 'roi.json')
                shutil.copy(source_roi, dst_video)
                count += 1

        print(f"[DataManager]       ROI distribuito in {count} cartelle video.")

    def prepare_behavior_gt(self):
        roi_path = self.cfg['paths']['roi']
        if not os.path.exists(roi_path):
            print(f"[DataManager] [ERROR] File ROI non trovato in {roi_path}")
            return

        with open(roi_path, 'r') as f:
            rois = json.load(f)

        subsets = ['train', 'test']

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if not os.path.exists(subset_path):
                continue

            video_folders = [f for f in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, f))]

            for video_name in tqdm(video_folders, desc=f"[DataManager] Gen Behavior {subset}", unit="vid"):
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
                    tqdm.write(f"[DataManager] [ERROR] Errore lettura GT per {video_name}: {e}")
                    continue

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

                            cx = row['x'] + (row['w'] / 2.0)
                            cy = row['y'] + (row['h'] / 2.0)

                            box_xywh = [cx, cy, row['w'], row['h']]

                            if GeometryUtils.is_in_roi(box_xywh, rois.get('roi1'), w_img, h_img):
                                counts[1] += 1
                            elif GeometryUtils.is_in_roi(box_xywh, rois.get('roi2'), w_img, h_img):
                                counts[2] += 1

                        f_out.write(f"{int(frame_id)},1,{counts[1]}\n")
                        f_out.write(f"{int(frame_id)},2,{counts[2]}\n")
                        f_out.flush()

    def _create_yolo_yaml(self):
        yaml_path = os.path.join(self.yolo_dataset_path, 'dataset.yaml')
        abs_path = os.path.abspath(self.yolo_dataset_path)

        data = {
            'path': abs_path,
            'train': 'images/train',
            'val': 'images/train',
            'test': 'images/test',
            'nc': 4,
            'names': {0: 'player_uf', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        }

        if not os.path.exists(os.path.join(abs_path, 'images', 'test')):
            if 'test' in data: del data['test']

        if os.path.exists(os.path.join(abs_path, 'images', 'valid')):
            data['val'] = 'images/valid'
        elif os.path.exists(os.path.join(abs_path, 'images', 'test')):
            data['val'] = 'images/test'

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"[DataManager]       YAML Creato: {os.path.basename(yaml_path)}")

    @staticmethod
    def _unzip_and_delete(source_folder, output_folder):
        if not os.path.exists(source_folder): return
        if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)

        found_zip = False
        for item in os.listdir(source_folder):
            if item.endswith(".zip"):
                found_zip = True
                zip_path = os.path.join(source_folder, item)
                print(f"[DataManager]       Estraendo: {item}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(output_folder)
                    os.remove(zip_path)
                except Exception as e:
                    print(f"[DataManager] [ERROR] Errore zip {item}: {e}")

        if not found_zip:
            print("[DataManager]       Nessun file .zip trovato, procedo.")

    def _convert_mot_to_yolo(self, source_dir, output_dir, sub_folder='train'):
        images_out = os.path.join(output_dir, 'images', sub_folder)
        labels_out = os.path.join(output_dir, 'labels', sub_folder)
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        video_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

        for video_name in tqdm(video_folders, desc=f"[DataManager] Convert {sub_folder}", unit="vid"):
            video_path = os.path.join(source_dir, video_name)

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

                img_name_base = f"{int(frame_id):06d}"
                src_img = os.path.join(img_dir, img_name_base + ".jpg")
                if not os.path.exists(src_img):
                    src_img = os.path.join(img_dir, img_name_base + ".png")
                if not os.path.exists(src_img): continue

                img = cv2.imread(src_img)
                if img is None: continue
                h_img, w_img = img.shape[:2]

                yolo_lines = []
                for _, row in group.iterrows():
                    obj_id = int(row['obj_id'])
                    label_str = id_to_label.get(obj_id, "unknown")

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

                    x_c = (row['x'] + row['w'] / 2) / w_img
                    y_c = (row['y'] + row['h'] / 2) / h_img
                    w_n = row['w'] / w_img
                    h_n = row['h'] / h_img

                    x_c = max(0, min(1, x_c))
                    y_c = max(0, min(1, y_c))
                    w_n = max(0, min(1, w_n))
                    h_n = max(0, min(1, h_n))

                    yolo_lines.append(f"{final_class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

                if yolo_lines:
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(yolo_lines))

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
        subsets = ['train', 'test']
        total_modified = 0
        total_skipped = 0
        total_errors = 0

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if not os.path.exists(subset_path):
                continue

            video_folders = [f for f in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, f))]

            for video_name in tqdm(video_folders, desc=f"[DataManager] Cleaning {subset}", unit="vid"):
                video_path = os.path.join(subset_path, video_name)
                gt_path = os.path.join(video_path, 'gt', 'gt.txt')
                backup_path = gt_path + ".bak"  # Percorso del backup
                ini_path = os.path.join(video_path, 'gameinfo.ini')

                if os.path.exists(backup_path):
                    total_skipped += 1
                    continue

                if not os.path.exists(gt_path) or not os.path.exists(ini_path):
                    continue

                try:
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

                    if new_count < original_count:
                        shutil.copy(gt_path, backup_path)
                        df_clean.to_csv(gt_path, header=False, index=False)
                        total_modified += 1
                    else:
                        pass

                except Exception as e:
                    tqdm.write(f"[DataManager] [ERROR] Errore su {video_name}: {e}")
                    total_errors += 1

        print(f"[DataManager]       Statistiche Pulizia:")
        print(f"[DataManager]       - Modificati: {total_modified}")
        print(f"[DataManager]       - Già Processati: {total_skipped}")
        if total_errors > 0:
            print(f"[DataManager]       - Errori: {total_errors}")