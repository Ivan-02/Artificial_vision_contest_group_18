import os
import zipfile
import shutil
import pandas as pd
import cv2
import yaml
from tqdm import tqdm


class DataManager:

    def __init__(self, config):
        self.cfg = config
        self.raw_data_path = self.cfg['paths']['raw_data']
        self.yolo_dataset_path = self.cfg['paths']['yolo_dataset']

    def prepare_dataset(self):
        """
        Pipeline principale:
        1. Estrae i zip se presenti.
        2. Converte le annotazioni da MOT a YOLO.
        3. Genera il file dataset.yaml per il training.
        """

        self._unzip_and_delete(self.raw_data_path, self.raw_data_path)

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
                    target_class_id=0,
                    sub_folder=subset
                )

        if not found_any:
            print("Nessuna sottocartella (train/test) trovata. Tento conversione nella root...")
            self._convert_mot_to_yolo(self.raw_data_path, self.yolo_dataset_path, sub_folder='train')

        # 3. Creazione file YAML per YOLO
        self._create_yolo_yaml()
        print("--- Preparazione Dataset Completata ---\n")

    def _create_yolo_yaml(self):
        yaml_path = os.path.join(self.yolo_dataset_path, 'dataset.yaml')
        abs_path = os.path.abspath(self.yolo_dataset_path)

        data = {
            'path': abs_path,
            'train': 'images/train',
            'val': 'images/train',
            'test': 'images/test'
        }

        if not os.path.exists(os.path.join(abs_path, 'images', 'test')):
            del data['test']

        # Se esiste validation separata, aggiorniamo
        if os.path.exists(os.path.join(abs_path, 'images', 'valid')):
            data['val'] = 'images/valid'

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)

        print(f"File di configurazione YOLO creato: {yaml_path}")


    def _unzip_and_delete(self, source_folder, output_folder):
        """
        Estrae tutti i file .zip dalla source_folder verso la output_folder
        e poi elimina i file .zip originali.
        """

        # 1. Controlla se la cartella sorgente esiste
        if not os.path.exists(source_folder):
            print(f"Errore: La cartella sorgente '{source_folder}' non esiste.")
            return

        # 2. Crea la cartella di destinazione se non esiste
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                print(f"Creata cartella di destinazione: {output_folder}")
            except OSError as e:
                print(f"Errore nella creazione della cartella {output_folder}: {e}")
                return

        # 3. Itera sui file
        files_found = False
        for item in os.listdir(source_folder):
            if item.endswith(".zip"):
                files_found = True
                zip_path = os.path.join(source_folder, item)

                print(f"Sto estraendo: {item} -> in {output_folder}")

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # Qui specifichiamo la cartella di output
                        zip_ref.extractall(output_folder)

                    # Se l'estrazione va a buon fine, elimina lo zip originale
                    print(f"Estrazione completata. Elimino {item}")
                    os.remove(zip_path)

                except zipfile.BadZipFile:
                    print(f"ERRORE: Il file {item} è corrotto.")
                except Exception as e:
                    print(f"Errore generico con {item}: {e}")

        if not files_found:
            print("Nessun file .zip trovato nella cartella sorgente.")
        else:
            print("\nOperazione completata!")


    def _convert_mot_to_yolo(self, source_dir, output_dir, target_class_id=0, sub_folder='train'):
        """
        Converte un dataset da formato MOT a formato YOLO.
        Salta l'elaborazione se i file di destinazione esistono già.
        """

        # 1. Creazione cartelle di output
        images_train_dir = os.path.join(output_dir, 'images', sub_folder)
        labels_train_dir = os.path.join(output_dir, 'labels', sub_folder)
        os.makedirs(images_train_dir, exist_ok=True)
        os.makedirs(labels_train_dir, exist_ok=True)

        # Trova tutte le cartelle dei video
        video_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

        print(f"Trovati {len(video_folders)} video da processare.")

        for video_name in tqdm(video_folders, desc=f"Processando {sub_folder}"):
            video_path = os.path.join(source_dir, video_name)

            # Percorsi specifici MOT
            det_path = os.path.join(video_path, 'det', 'det.txt')
            img_dir = os.path.join(video_path, 'img1')

            # Controllo esistenza file fondamentali
            if not os.path.exists(det_path) or not os.path.exists(img_dir):
                continue

            # 2. Leggi il file det.txt
            try:
                df = pd.read_csv(det_path, header=None)
                df.columns = ['frame_id', 'obj_id', 'x_topleft', 'y_topleft', 'width', 'height', 'conf', 'class_id', 'vis',
                              'unused']
            except Exception as e:
                print(f"Errore lettura CSV per {video_name}: {e}")
                continue

            # 3. Raggruppa per Frame ID
            grouped = df.groupby('frame_id')

            for frame_id, group in grouped:
                # Costruisci il nome univoco SUBITO, per controllare se esiste
                unique_name = f"{video_name}_frame_{int(frame_id):06d}"
                label_path_dest = os.path.join(labels_train_dir, unique_name + '.txt')

                # --- CONTROLLO ESISTENZA ---
                # Se il file label esiste già, saltiamo tutto il blocco pesante (lettura img, calcoli, scrittura)
                if os.path.exists(label_path_dest):
                    continue
                # ---------------------------

                # Costruisci il nome del file immagine sorgente (es. 000001.jpg)
                file_img_name = f"{int(frame_id):06d}.jpg"
                path_img_src = os.path.join(img_dir, file_img_name)

                # Fallback se le immagini sono png
                if not os.path.exists(path_img_src):
                    file_img_name = f"{int(frame_id):06d}.png"
                    path_img_src = os.path.join(img_dir, file_img_name)

                if not os.path.exists(path_img_src):
                    continue  # Immagine non trovata, salto

                # 4. Ottieni dimensioni immagine (necessario per normalizzare)
                # cv2.imread è lento, quindi lo facciamo solo se non abbiamo saltato sopra
                img = cv2.imread(path_img_src)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                # 5. Prepara il file label YOLO
                yolo_labels = []

                for _, row in group.iterrows():
                    # Calcolo centro assoluto e normalizzazione
                    x_center_abs = row['x_topleft'] + (row['width'] / 2)
                    y_center_abs = row['y_topleft'] + (row['height'] / 2)

                    x_center_norm = x_center_abs / img_w
                    y_center_norm = y_center_abs / img_h
                    width_norm = row['width'] / img_w
                    height_norm = row['height'] / img_h

                    # Clipping (0-1)
                    x_center_norm = min(max(x_center_norm, 0), 1)
                    y_center_norm = min(max(y_center_norm, 0), 1)
                    width_norm = min(max(width_norm, 0), 1)
                    height_norm = min(max(height_norm, 0), 1)

                    # Scrittura riga YOLO: class x y w h
                    yolo_line = f"{target_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                    yolo_labels.append(yolo_line)

                # 6. Salvataggio Label e Collegamento Immagine
                if yolo_labels:
                    # A. Salva il file .txt
                    with open(label_path_dest, 'w') as f:
                        f.write('\n'.join(yolo_labels))

                    # B. Crea SYMLINK per l'immagine (o copia se fallisce)
                    img_ext = os.path.splitext(file_img_name)[1]
                    img_path_dest = os.path.join(images_train_dir, unique_name + img_ext)

                    # Ottieni percorsi assoluti (NECESSARIO per i symlink)
                    src_abs = os.path.abspath(path_img_src)
                    dst_abs = os.path.abspath(img_path_dest)

                    # Rimuovi eventuale link rotto o file esistente per ricrearlo pulito (solo se siamo arrivati qui)
                    if os.path.exists(dst_abs) or os.path.islink(dst_abs):
                        try:
                            os.remove(dst_abs)
                        except OSError:
                            pass

                    try:
                        # Tenta di creare il collegamento simbolico
                        os.symlink(src_abs, dst_abs)
                    except OSError:
                        # Se l'OS non permette i symlink (es. Windows senza permessi Admin), facciamo fallback sulla copia
                        shutil.copy(src_abs, dst_abs)

        print("\nConversione completata!")
        print(f"Dataset YOLO pronto in: {output_dir}")