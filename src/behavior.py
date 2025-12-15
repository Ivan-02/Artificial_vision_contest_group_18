import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from .tracker import Tracker


class BehaviorAnalyzer:
    def __init__(self, config, conf_mode):
        self.cfg = config
        self.tracker = Tracker(config, conf_mode)
        self.output_dir = os.path.join(self.cfg['paths']['output_submission'], conf_mode['test_name'], "_behavior")
        os.makedirs(self.output_dir, exist_ok=True)

        # Definizione colori (BGR)
        self.colors = {
            'default': (0, 0, 255),  # Rosso per giocatori fuori ROI
            'roi1': (0, 255, 255),  # Giallo
            'roi2': (255, 0, 255)  # Magenta
        }

    def _load_rois(self, video_folder):
        """Legge il file JSON delle ROI."""
        json_path = self.cfg['paths']['roi']

        if not os.path.exists(json_path):
            # Fallback dummy se non trovato (utile per debug senza i file reali)
            return {"roi1": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.4},
                    "roi2": {"x": 0.5, "y": 0.7, "width": 0.5, "height": 0.3}}

        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def _is_in_roi(self, box_xywh, roi, img_w, img_h):
        """
        Verifica se il giocatore è nella ROI secondo la logica del PDF.
        Logica: "A player is considered in a ROI if the center of the basis
        [cite_start]of the bounding box is inside the ROI" [cite: 98]
        """
        x_c, y_c, w_box, h_box = box_xywh

        # [cite_start]Calcolo Centro della Base (Center of the basis) [cite: 98]
        base_x = x_c
        base_y = y_c + (h_box / 2.0)

        # ROI coordinates (relative [0,1]) -> Absolute
        # [cite_start]Il PDF dice: absolute = relative * image_size [cite: 97]
        roi_x = roi['x'] * img_w
        roi_y = roi['y'] * img_h
        roi_w = roi['width'] * img_w
        roi_h = roi['height'] * img_h

        # Controllo se il punto è nel rettangolo
        in_x = roi_x <= base_x <= (roi_x + roi_w)
        in_y = roi_y <= base_y <= (roi_y + roi_h)

        return in_x and in_y

    def run(self):
        video_folders = self.tracker.get_video_list()
        stop_execution = False  # Flag per l'uscita anticipata con 'q'

        for video_name in tqdm(video_folders, desc="Behavior Analysis"):
            if stop_execution: break

            video_id = video_name.split('-')[1].split('.')[0]

            # Definisco il nome della finestra qui per usarlo sia nel display che nella chiusura
            window_name = f"Behavior - {video_name}"

            # Carica ROI
            rois_data = self._load_rois(video_name)
            # Se rois_data è None o vuoto, usa un fallback o gestisci l'errore
            if not rois_data:
                rois_data = {"roi1": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.4},
                             "roi2": {"x": 0.5, "y": 0.7, "width": 0.5, "height": 0.3}}

            output_file = os.path.join(self.output_dir, f"behavior_{video_id}_{self.cfg['names']['team']}.txt")

            if os.path.exists(output_file):
                print(f"\nSkipping {video_name}, file esistente.")
                continue

            with open(output_file, 'w') as f_out:
                # Usa il generatore del Tracker
                for frame_id, img, detections in self.tracker.track_video_generator(video_name):
                    h_img, w_img = img.shape[:2]

                    count_roi1 = 0
                    count_roi2 = 0

                    # Prima passata: Calcolo conteggi e determinazione colori
                    # Creiamo una lista di tuple (detection, colore_assegnato) da disegnare dopo
                    to_draw = []

                    for det in detections:
                        # Colore di default: Rosso (fuori ROI)
                        assigned_color = self.colors['default']
                        is_inside = False

                        # Controllo ROI 1
                        if self._is_in_roi(det['xywh'], rois_data.get('roi1'), w_img, h_img):
                            count_roi1 += 1
                            assigned_color = self.colors['roi1']
                            is_inside = True

                        # Controllo ROI 2 (Se è in entrambe, prevale l'ultima controllata o logica custom)
                        elif self._is_in_roi(det['xywh'], rois_data.get('roi2'), w_img, h_img):
                            count_roi2 += 1
                            assigned_color = self.colors['roi2']
                            is_inside = True

                        to_draw.append((det, assigned_color))

                    # Gestione Visualizzazione
                    if self.tracker.enable_display:
                        # 1. Disegna le ROI con i contatori aggiornati
                        current_counts = {'roi1': count_roi1, 'roi2': count_roi2}
                        self._draw_rois_with_counts(img, rois_data, w_img, h_img, current_counts)

                        # 2. Disegna i giocatori con i colori corretti
                        for det, color in to_draw:
                            self._draw_player_box(img, det, color)

                        # Mostra frame e gestisci tasto 'q'
                        # tracker._show_frame ritorna False se 'q' è premuto
                        if not self.tracker._show_frame(window_name, img):
                            print("\nInterruzione richiesta dall'utente (Tasto Q).")
                            stop_execution = True
                            break

                    # Scrittura output su file
                    f_out.write(f"{frame_id},1,{count_roi1}\n")
                    f_out.write(f"{frame_id},2,{count_roi2}\n")
                    f_out.flush()

            # --- MODIFICA QUI ---
            # Una volta finito il video corrente (usciti dal blocco 'with open' e dal loop dei frame),
            # chiudiamo la finestra specifica di questo video.
            if self.tracker.enable_display:
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    pass  # Ignora errore se la finestra è già chiusa o non esisteva

        if self.tracker.enable_display:
            cv2.destroyAllWindows()

        print(f"Behavior Analysis Completata. File salvati in {self.output_dir}")

    def _draw_rois_with_counts(self, img, rois, w, h, counts):
        """
        Disegna i rettangoli delle ROI e un box informativo col conteggio sopra di esse.
        """
        for i, (key, r) in enumerate(rois.items()):
            # Determina colore ROI
            color = self.colors.get(key, (255, 255, 255))

            # Coordinate ROI
            x = int(r['x'] * w)
            y = int(r['y'] * h)
            rw = int(r['width'] * w)
            rh = int(r['height'] * h)

            # 1. Disegna rettangolo ROI
            cv2.rectangle(img, (x, y), (x + rw, y + rh), color, 2)

            # 2. Box Conteggio sopra la ROI
            count_val = counts.get(key, 0)
            label_text = f"{key.upper()}: {count_val}"

            # Calcola dimensioni testo per background
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            # Coordinate per il box del conteggio (sopra la ROI)
            box_x1 = x
            box_y1 = y - text_h - 10
            box_x2 = x + text_w + 10
            box_y2 = y

            # Assicuriamoci di non uscire dall'immagine in alto
            if box_y1 < 0:
                box_y1 = y
                box_y2 = y + text_h + 10

            # Disegna sfondo del label (pieno)
            cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), color, -1)

            # Scrivi testo (Nero o Bianco a seconda del contrasto, qui usiamo Nero per visibilità su Giallo/Magenta)
            text_color = (0, 0, 0)
            cv2.putText(img, label_text, (box_x1 + 5, box_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    def _draw_player_box(self, img, det, color):
        """
        Disegna bounding box, label ID e confidenza.
        """
        x1, y1, x2, y2 = map(int, det['xyxy'])

        # 1. Bounding Box del giocatore
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 2. Punto alla base (per debug logica ROI)
        base_x = int(det['xywh'][0])
        base_y = int(det['xywh'][1] + det['xywh'][3] / 2)
        cv2.circle(img, (base_x, base_y), 4, color, -1)

        # 3. Label (ID e Confidenza)
        track_id = det['track_id']
        conf = det['conf']
        label = f"ID:{track_id} {conf:.2f}"

        # Sfondo per il testo
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w_text, y1), color, -1)

        # Testo (Bianco o Nero)
        text_color = (255, 255, 255) if color == (0, 0, 255) else (0, 0, 0)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)