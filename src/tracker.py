import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import traceback


class Tracker:
    def __init__(self, config):
        self.cfg = config

        # Carica il modello
        print("Caricamento modello...")
        self.model = YOLO(self.cfg['paths']['model_weights'])

        # Verifica classi del modello (Debug)
        print(f"Classi del modello: {self.model.names}")
        # Dovrebbe essere {0: 'ball', 1: 'player', 2: 'referee', ...}

        self.output_dir = self.cfg['paths']['output_submission']
        os.makedirs(self.output_dir, exist_ok=True)

        self.conf = self.cfg['tracking']['conf_threshold']
        self.iou = self.cfg['tracking']['iou_threshold']

        self.tracker_yaml_path = self.cfg['paths'].get('temp_tracker_yaml', './temp_tracker.yaml')

        # --- IMPOSTAZIONI VISUALIZZAZIONE ---
        self.display_width = 1280
        self.enable_display = True
        # ------------------------------------

        # Generazione file config temporaneo per il tracker
        tracker_settings = self.cfg['tracking']['tracker_settings']
        os.makedirs(os.path.dirname(self.tracker_yaml_path), exist_ok=True)
        with open(self.tracker_yaml_path, 'w') as f:
            yaml.dump(tracker_settings, f, sort_keys=False)

        print(f"File configurazione tracker generato in: {self.tracker_yaml_path}")

    def _get_id_color(self, id_int):
        """Genera un colore BGR consistente per ogni ID."""
        np.random.seed(int(id_int))
        color = np.random.randint(0, 255, size=3).tolist()
        return tuple(color)

    def run(self):
        print(f"--- Inizio Tracking ---")
        if self.enable_display:
            print("Visualizzazione attiva: Premi 'q' sulla finestra video per interrompere.")

        # Costruzione percorso dati
        test_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['val']['split'])

        # Controllo di sicurezza esistenza cartella
        if not os.path.exists(test_dir):
            print(f"ERRORE CRITICO: La cartella dati non esiste: {test_dir}")
            return

        video_folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

        if not video_folders:
            print(f"Nessuna cartella video trovata in {test_dir}")
            return

        # Loop principale sui video
        for video_name in tqdm(video_folders, desc="Tracking Videos"):
            video_path = os.path.join(test_dir, video_name, 'img1')

            # Controllo esistenza immagini per evitare warning 'source is missing'
            if not os.path.exists(video_path):
                print(f"SKIP: Percorso immagini non trovato per {video_name}")
                continue

            output_txt = os.path.join(self.output_dir, f"{video_name}.txt")

            if os.path.exists(output_txt):
                print(f"Skipping {video_name}, file esistente.")
                continue

            # Tracking
            # show=False perché gestiamo noi la visualizzazione per evitare errori
            results = self.model.track(
                source=video_path,
                conf=self.conf,
                iou=self.iou,
                tracker=self.tracker_yaml_path,
                classes=[1, 2, 3],  # Chiediamo esplicitamente di ignorare la 0
                persist=True,
                device=self.cfg['tracking']['tracker_settings']['device'],
                verbose=False,
                stream=True,
                show=False,
                agnostic_nms=True,
                imgsz=self.cfg['tracking']['tracker_settings']['imgsz'],
                half=True,
            )

            f_out = open(output_txt, 'w')
            window_name = f"Monitor - {video_name}"
            window_created = False

            try:
                # Loop sui frame
                for result in results:
                    # Estrazione ID frame dal nome file
                    if hasattr(result, 'path'):
                        try:
                            frame_idx = os.path.basename(result.path).split('.')[0]
                            frame_id = int(frame_idx)
                        except ValueError:
                            frame_id = 0
                    else:
                        continue

                    # Copia immagine per display
                    if self.enable_display:
                        frame_display = result.orig_img.copy()

                    # Se ci sono detection
                    if result.boxes.id is not None:
                        # Estrazione dati tensori -> numpy
                        boxes_xywh = result.boxes.xywh.cpu().numpy()
                        track_ids = result.boxes.id.int().cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        cls_ids = result.boxes.cls.int().cpu().numpy()  # Le classi predette

                        # --- VERIFICA CLASSI ---
                        # Se per sbaglio passa una classe 0, la filtriamo manualmente qui
                        if 0 in cls_ids:
                            pass

                        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)

                        for i, track_id in enumerate(track_ids):
                            current_class = cls_ids[i]

                            # DOPPIA SICUREZZA: Se è palla (0), saltiamo tutto
                            if current_class == 0:
                                continue

                            conf = confs[i]

                            # 1. Scrittura su file (xywh) per la sottomissione
                            x_c, y_c, w, h = boxes_xywh[i]
                            x1_txt = x_c - (w / 2)
                            y1_txt = y_c - (h / 2)
                            line = f"{frame_id},{track_id},{x1_txt:.2f},{y1_txt:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
                            f_out.write(line)

                            # 2. Disegno (MODIFICATO: Solo ID box in basso al centro)
                            if self.enable_display:
                                color_bgr = self._get_id_color(track_id)
                                x1_draw, y1_draw, x2_draw, y2_draw = boxes_xyxy[i]

                                # --- LOGICA DI VISUALIZZAZIONE PULITA ---

                                # Calcolo il centro della base del box (i piedi del giocatore)
                                center_x = int((x1_draw + x2_draw) / 2)
                                bottom_y = int(y2_draw)

                                # Preparazione testo (Solo ID)
                                label_text = str(track_id)
                                font_face = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                thickness = 2

                                # Calcolo dimensioni del testo per creare lo sfondo su misura
                                (text_w, text_h), baseline = cv2.getTextSize(label_text, font_face, font_scale,
                                                                             thickness)

                                # Padding (margine) attorno al numero
                                pad_x = 8
                                pad_y = 6

                                box_w = text_w + (pad_x * 2)
                                box_h = text_h + (pad_y * 2)

                                # Coordinate del piccolo riquadro ID
                                # Lo posizioniamo centrato orizzontalmente rispetto al giocatore
                                # e appena sopra la linea dei piedi (bottom_y)
                                box_x1 = center_x - (box_w // 2)
                                box_y1 = bottom_y - box_h
                                box_x2 = box_x1 + box_w
                                box_y2 = bottom_y

                                # Disegno sfondo colorato (pieno)
                                cv2.rectangle(frame_display, (box_x1, box_y1), (box_x2, box_y2), color_bgr, -1)

                                # Disegno bordo bianco sottile per contrasto
                                cv2.rectangle(frame_display, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 1)

                                # Scrittura del numero (Bianco per essere leggibile sul colore)
                                text_x = box_x1 + pad_x
                                text_y = box_y1 + text_h + pad_y - 2  # Aggiustamento fine per centratura verticale

                                cv2.putText(frame_display, label_text, (text_x, text_y),
                                            font_face, font_scale, (255, 255, 255), thickness)

                    f_out.flush()

                    # --- GESTIONE FINESTRA ---
                    if self.enable_display:
                        h_orig, w_orig = frame_display.shape[:2]
                        if h_orig > 0 and w_orig > 0:
                            # Resize intelligente
                            aspect_ratio = h_orig / w_orig
                            display_height = int(self.display_width * aspect_ratio)
                            frame_resized = cv2.resize(frame_display, (self.display_width, display_height))

                            cv2.imshow(window_name, frame_resized)
                            window_created = True

                            # Tasto Q per uscire
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("\nInterruzione richiesta dall'utente.")
                                f_out.close()
                                cv2.destroyAllWindows()
                                return

            except Exception as e:
                print(f"Errore durante il video {video_name}: {e}")
                traceback.print_exc()

            finally:
                f_out.close()
                if self.enable_display and window_created:
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass

        if self.enable_display:
            cv2.destroyAllWindows()
        print("\n--- Tracking Completato ---")