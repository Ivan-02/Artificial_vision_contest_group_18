import os
import cv2
import yaml
import numpy as np  # Necessario per generare i colori
from ultralytics import YOLO
from tqdm import tqdm


class Tracker:
    def __init__(self, config):
        self.cfg = config
        self.model = YOLO(self.cfg['paths']['model_weights'])

        self.output_dir = self.cfg['paths']['output_submission']
        os.makedirs(self.output_dir, exist_ok=True)

        self.conf = self.cfg['tracking']['conf_threshold']
        self.iou = self.cfg['tracking']['iou_threshold']

        self.tracker_yaml_path = self.cfg['paths'].get('temp_tracker_yaml', './temp_tracker.yaml')

        # --- IMPOSTAZIONI VISUALIZZAZIONE ---
        # Larghezza desiderata per la finestra video
        self.display_width = 1280
        self.enable_display = True  # Metti False se vuoi tornare alla modalità veloce senza video
        # ------------------------------------

        tracker_settings = self.cfg['tracking']['tracker_settings']
        with open(self.tracker_yaml_path, 'w') as f:
            yaml.dump(tracker_settings, f, sort_keys=False)

        print(f"File configurazione tracker generato in: {self.tracker_yaml_path}")

    def _get_id_color(self, id_int):
        """
        Genera un colore BGR unico e consistente basato su un numero intero (ID).
        Usa l'ID come seed per la randomizzazione, così lo stesso ID ha sempre lo stesso colore.
        """
        np.random.seed(id_int)
        # Genera 3 numeri casuali tra 0 e 255 (BGR per OpenCV)
        color = np.random.randint(0, 255, size=3).tolist()
        return tuple(color)

    def run(self):
        print(f"--- Inizio Tracking ---")
        if self.enable_display:
            print("Visualizzazione attiva: Premi 'q' sulla finestra video per interrompere.")

        test_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['val']['split'])
        video_folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

        # Loop principale sui video
        for video_name in tqdm(video_folders, desc="Tracking Videos"):
            video_path = os.path.join(test_dir, video_name, 'img1')
            output_txt = os.path.join(self.output_dir, f"{video_name}.txt")

            if os.path.exists(output_txt):
                print(f"Skipping {video_name}, file esistente.")
                continue

            # Importante: show=False perché gestiamo noi la visualizzazione
            results = self.model.track(
                source=video_path,
                conf=self.conf,
                iou=self.iou,
                tracker=self.tracker_yaml_path,
                classes=[1, 2, 3],
                persist=True,
                device=self.cfg['training']['device'],
                verbose=False,
                stream=True,
                show=True
            )

            # Apriamo il file fuori dal ciclo dei frame per efficienza
            f_out = open(output_txt, 'w')

            window_name = f"Monitor Tracking - {video_name}"

            try:
                # Loop sui frame del video corrente
                for result in results:
                    # Gestione frame ID
                    if hasattr(result, 'path'):
                        frame_idx = os.path.basename(result.path).split('.')[0]
                        frame_id = int(frame_idx)
                    else:
                        continue

                    # --- PREPARAZIONE VISUALIZZAZIONE (Copia dell'immagine originale) ---
                    if self.enable_display:
                        frame_display = result.orig_img.copy()

                    # Se ci sono detection valide con ID
                    if result.boxes.id is not None:
                        # Dati per il file di testo (xywh float)
                        boxes_xywh = result.boxes.xywh.cpu().numpy()
                        track_ids = result.boxes.id.int().cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()

                        # Dati per il disegno OpenCV (xyxy interi)
                        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)

                        for i, track_id in enumerate(track_ids):
                            conf = confs[i]

                            # --- 1. SCRITTURA SU FILE (Logica originale) ---
                            x_c, y_c, w, h = boxes_xywh[i]
                            x1_txt = x_c - (w / 2)
                            y1_txt = y_c - (h / 2)
                            line = f"{frame_id},{track_id},{x1_txt:.2f},{y1_txt:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
                            f_out.write(line)

                            # --- 2. DISEGNO SU FRAME (Nuova logica) ---
                            if self.enable_display:
                                # Ottieni colore unico per questo ID
                                color_bgr = self._get_id_color(track_id)
                                x1_draw, y1_draw, x2_draw, y2_draw = boxes_xyxy[i]

                                # Disegna rettangolo
                                cv2.rectangle(frame_display, (x1_draw, y1_draw), (x2_draw, y2_draw), color_bgr, 2)

                                # Disegna etichetta (ID e Confidenza)
                                label = f"ID:{track_id} ({conf:.2f})"
                                # Sfondo per il testo per renderlo leggibile
                                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                cv2.rectangle(frame_display, (x1_draw, y1_draw - 20), (x1_draw + w_text, y1_draw),
                                              color_bgr, -1)
                                cv2.putText(frame_display, label, (x1_draw, y1_draw - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    f_out.flush()  # Scrive su disco per ogni frame per sicurezza

                    # --- GESTIONE FINESTRA VIDEO ---
                    if self.enable_display:
                        # Calcolo resize mantenendo aspect ratio
                        h_orig, w_orig = frame_display.shape[:2]
                        aspect_ratio = h_orig / w_orig
                        display_height = int(self.display_width * aspect_ratio)

                        # Ridimensiona
                        frame_resized = cv2.resize(frame_display, (self.display_width, display_height))

                        # Mostra
                        cv2.imshow(window_name, frame_resized)

                        # Gestione tasto 'q' per uscire anticipatamente
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nInterruzione richiesta dall'utente.")
                            f_out.close()
                            cv2.destroyAllWindows()
                            return  # Esce completamente dalla funzione run()

            finally:
                # Assicura che il file venga chiuso e la finestra del video corrente distrutta
                f_out.close()
                if self.enable_display:
                    cv2.destroyWindow(window_name)

        if self.enable_display:
            cv2.destroyAllWindows()
        print("\n--- Tracking Completato ---")