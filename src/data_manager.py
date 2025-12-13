import os
import zipfile
import shutil
import pandas as pd
import cv2
import yaml
import configparser
from tqdm import tqdm


class DataManager:

    def __init__(self, config):
        self.cfg = config
        self.raw_data_path = self.cfg['paths']['raw_data']
        self.yolo_dataset_path = self.cfg['paths']['yolo_dataset']

    def prepare_dataset(self):
        """
        Pipeline: Unzip -> Converti (filtrando palla e assegnando ID corretti) -> Crea YAML
        """
        # 1. Estrazione
        self._unzip_and_delete(self.raw_data_path, self.raw_data_path)

        # 2. Conversione
        subsets = ['train', 'test']
        found_any = False

        for subset in subsets:
            subset_path = os.path.join(self.raw_data_path, subset)
            if os.path.exists(subset_path):
                print(f"Trovato subset: {subset}. Inizio conversione...")
                found_any = True
                # NOTA: Non passo più target_class_id perché lo decidiamo dinamicamente dentro
                self._convert_mot_to_yolo(
                    source_dir=subset_path,
                    output_dir=self.yolo_dataset_path,
                    sub_folder=subset
                )

        if not found_any:
            print("Nessuna sottocartella standard trovata. Tento conversione nella root...")
            self._convert_mot_to_yolo(self.raw_data_path, self.yolo_dataset_path, sub_folder='train')

        # 3. Creazione file YAML
        self._create_yolo_yaml()
        print("--- Preparazione Dataset Completata ---\n")

    def _create_yolo_yaml(self):
        yaml_path = os.path.join(self.yolo_dataset_path, 'dataset.yaml')
        abs_path = os.path.abspath(self.yolo_dataset_path)

        # DEFINIAMO LE CLASSI COME LE VUOLE IL MODELLO (ID 0..3)
        # Anche se noi non scriveremo mai '0' nei file txt, il modello ha 4 output.
        data = {
            'path': abs_path,
            'train': 'images/train',
            'val': 'images/train',
            'test': 'images/test',
            'nc': 4,
            'names': {
                0: 'ball',
                1: 'goalkeeper',  # Nei txt sarà classe 1
                2: 'player',  # Nei txt sarà classe 2
                3: 'referee'  # Nei txt sarà classe 3
            }
        }

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
        if not os.path.exists(source_folder):
            return
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

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

    @staticmethod
    def _parse_gameinfo(ini_path):
        """ Legge gameinfo.ini per mappare ID locale -> etichetta (es. "player team left") """
        id_map = {}
        if not os.path.exists(ini_path):
            return id_map
        try:
            config = configparser.ConfigParser()
            config.read(ini_path)
            if 'Sequence' in config:
                for key, val in config['Sequence'].items():
                    if key.startswith('trackletid_'):
                        try:
                            # key format: trackletid_X
                            obj_id = int(key.split('_')[1])
                            # val format: label; info
                            label_desc = val.split(';')[0].lower().strip()
                            id_map[obj_id] = label_desc
                        except:
                            continue
        except Exception:
            pass
        return id_map

    def _convert_mot_to_yolo(self, source_dir, output_dir, sub_folder='train'):
        images_out = os.path.join(output_dir, 'images', sub_folder)
        labels_out = os.path.join(output_dir, 'labels', sub_folder)
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        video_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

        for video_name in tqdm(video_folders, desc=f"Processando {sub_folder}"):
            video_path = os.path.join(source_dir, video_name)

            # 1. Recupera mappatura ID -> Label (es. 1->gk, 20->ball)
            ini_path = os.path.join(video_path, 'gameinfo.ini')
            id_to_label = self._parse_gameinfo(ini_path)

            # 2. Usa GT.txt (Ground Truth)
            gt_path = os.path.join(video_path, 'gt', 'gt.txt')
            img_dir = os.path.join(video_path, 'img1')

            if not os.path.exists(gt_path) or not os.path.exists(img_dir):
                continue

            try:
                df = pd.read_csv(gt_path, header=None)
                # Colonne: frame, id, x, y, w, h, ...
                df.columns = ['frame_id', 'obj_id', 'x', 'y', 'w', 'h', 'conf', 'cls', 'vis', 'u']
            except:
                continue

            grouped = df.groupby('frame_id')

            for frame_id, group in grouped:
                unique_name = f"{video_name}_frame_{int(frame_id):06d}"
                label_file = os.path.join(labels_out, unique_name + '.txt')

                # Salta se esiste già
                if os.path.exists(label_file):
                    continue

                # Gestione immagine (jpg/png)
                img_name_base = f"{int(frame_id):06d}"
                src_img = os.path.join(img_dir, img_name_base + ".jpg")
                if not os.path.exists(src_img):
                    src_img = os.path.join(img_dir, img_name_base + ".png")
                if not os.path.exists(src_img):
                    continue

                # Leggi dimensioni per normalizzare
                img = cv2.imread(src_img)
                if img is None: continue
                h_img, w_img = img.shape[:2]

                yolo_lines = []
                for _, row in group.iterrows():
                    obj_id = int(row['obj_id'])

                    # Recupera l'etichetta testuale (es. "ball", "goalkeeper")
                    label_str = id_to_label.get(obj_id, "unknown")

                    # --- LOGICA DI FILTRO E ASSEGNAZIONE ID ---
                    final_class_id = -1

                    if "ball" in label_str:
                        final_class_id = 0
                    elif "goalkeeper" in label_str:
                        final_class_id = 1
                    elif "player" in label_str:
                        final_class_id = 2
                    elif "referee" in label_str:
                        final_class_id = 3

                    # Se non abbiamo assegnato un ID valido (es. unknown o altro), saltiamo
                    if final_class_id == -1:
                        continue

                    # Calcoli YOLO
                    x_c = (row['x'] + row['w'] / 2) / w_img
                    y_c = (row['y'] + row['h'] / 2) / h_img
                    w_n = row['w'] / w_img
                    h_n = row['h'] / h_img

                    # Clipping 0-1
                    x_c = max(0, min(1, x_c))
                    y_c = max(0, min(1, y_c))
                    w_n = max(0, min(1, w_n))
                    h_n = max(0, min(1, h_n))

                    yolo_lines.append(f"{final_class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

                # Scrivi file solo se ci sono oggetti validi
                if yolo_lines:
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(yolo_lines))

                    # Link immagine
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