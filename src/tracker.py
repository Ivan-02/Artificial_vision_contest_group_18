import os
import sys
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import traceback


class Tracker:
    def __init__(self, config, conf_mode):
        self.cfg = config
        self.cfg_mode = conf_mode

        self._check_keys()

        print("Caricamento modello...")
        self.model = YOLO(self.cfg['paths']['model_weights'])

        print(f"Classi del modello: {self.model.names}")

        self.output_dir = os.path.join(self.cfg['paths']['output_submission'], self.cfg_mode['test_name'])
        os.makedirs(self.output_dir, exist_ok=True)

        self.conf = self.cfg_mode['conf_threshold']
        self.iou = self.cfg_mode['iou_threshold']

        self.tracker_yaml_path = self.cfg['paths'].get('temp_tracker_yaml', './temp_tracker.yaml')

        self.display_width = self.cfg_mode['display_width']
        self.enable_display = self.cfg_mode['display']

        tracker_settings = self.cfg_mode['tracker_settings']
        os.makedirs(os.path.dirname(self.tracker_yaml_path), exist_ok=True)

        with open(self.tracker_yaml_path, 'w') as f:
            yaml.dump(tracker_settings, f, sort_keys=False)

        print(f"File configurazione tracker generato in: {self.tracker_yaml_path}")



    def _check_keys(self):
        required_keys = {
            'test_name','conf_threshold', 'iou_threshold', 'display', 'display_width',
            'half', 'show', 'imgsz', 'stream', 'verbose', 'persist',
            'agnostic_nms', 'classes', 'tracker_settings'
        }

        required_tracker_keys = {
            'tracker_type', 'track_high_thresh', 'track_low_thresh',
            'new_track_thresh', 'track_buffer', 'match_thresh',
            'gmc_method', 'proximity_thresh', 'appearance_thresh',
            'with_reid', 'fuse_score', 'reid_model', 'model'
        }

        current_keys = set(self.cfg_mode.keys())
        missing_keys = required_keys - current_keys

        if missing_keys:
            print(f"[ERRORE CONFIG] Mancano le seguenti chiavi principali: {list(missing_keys)}")
            sys.exit(1)

        current_tracker_keys = set(self.cfg_mode['tracker_settings'].keys())
        missing_tracker_keys = required_tracker_keys - current_tracker_keys

        if missing_tracker_keys:
            print(f"\n[ERRORE CONFIG] In 'tracker_settings' mancano: {list(missing_tracker_keys)}")
            sys.exit(1)


    @staticmethod
    def _get_id_color(id_int):
        np.random.seed(int(id_int))
        color = np.random.randint(0, 255, size=3).tolist()
        return tuple(color)

    def run(self):
        print(f"--- Inizio Tracking ---")
        if self.enable_display:
            print("Visualizzazione attiva: Premi 'q' sulla finestra video per interrompere.")

        test_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['paths']['split'])

        if not os.path.exists(test_dir):
            print(f"ERRORE CRITICO: La cartella dati non esiste: {test_dir}")
            return

        video_folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

        if not video_folders:
            print(f"Nessuna cartella video trovata in {test_dir}")
            return

        for video_name in tqdm(video_folders, desc="Tracking Videos"):
            video_path = os.path.join(test_dir, video_name, 'img1')

            if not os.path.exists(video_path):
                print(f"SKIP: Percorso immagini non trovato per {video_name}")
                continue

            output_txt = os.path.join(self.output_dir, f"{video_name}.txt")

            if os.path.exists(output_txt):
                print(f"Skipping {video_name}, file esistente.")
                continue

            results = self.model.track(
                source=video_path,
                conf=self.conf,
                iou=self.iou,
                tracker=self.tracker_yaml_path,
                classes=self.cfg_mode['classes'],
                persist=self.cfg_mode['persist'],
                device=self.cfg['device'],
                verbose=self.cfg_mode['verbose'],
                stream=self.cfg_mode['stream'],
                show=self.cfg_mode['show'],
                agnostic_nms=self.cfg_mode['agnostic_nms'],
                imgsz=self.cfg_mode['imgsz'],
                half=self.cfg_mode['half'],
            )

            f_out = open(output_txt, 'w')
            window_name = f"Monitor - {video_name}"
            window_created = False

            try:
                for result in results:
                    if hasattr(result, 'path'):
                        try:
                            frame_idx = os.path.basename(result.path).split('.')[0]
                            frame_id = int(frame_idx)
                        except ValueError:
                            frame_id = 0
                    else:
                        continue

                    if self.enable_display:
                        frame_display = result.orig_img.copy()

                    if result.boxes.id is not None:
                        boxes_xywh = result.boxes.xywh.cpu().numpy()
                        track_ids = result.boxes.id.int().cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        cls_ids = result.boxes.cls.int().cpu().numpy()

                        if 0 in cls_ids:
                            pass

                        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)

                        for i, track_id in enumerate(track_ids):
                            current_class = cls_ids[i]

                            if current_class == 0:
                                continue

                            conf = confs[i]

                            x_c, y_c, w, h = boxes_xywh[i]
                            x1_txt = x_c - (w / 2)
                            y1_txt = y_c - (h / 2)
                            line = f"{frame_id},{track_id},{x1_txt:.2f},{y1_txt:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
                            f_out.write(line)

                            if self.enable_display:
                                color_bgr = self._get_id_color(track_id)
                                x1_draw, y1_draw, x2_draw, y2_draw = boxes_xyxy[i]

                                cv2.rectangle(frame_display, (x1_draw, y1_draw), (x2_draw, y2_draw), color_bgr, 2)

                                label = f"ID:{track_id} ({conf:.2f})"

                                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                cv2.rectangle(frame_display, (x1_draw, y1_draw - 20), (x1_draw + w_text, y1_draw),
                                              color_bgr, -1)
                                cv2.putText(frame_display, label, (x1_draw, y1_draw - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    f_out.flush()

                    if self.enable_display:
                        h_orig, w_orig = frame_display.shape[:2]
                        if h_orig > 0 and w_orig > 0:
                            aspect_ratio = h_orig / w_orig
                            display_height = int(self.display_width * aspect_ratio)
                            frame_resized = cv2.resize(frame_display, (self.display_width, display_height))

                            cv2.imshow(window_name, frame_resized)
                            window_created = True

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nInterruzione richiesta dall'utente.")
                            f_out.close()
                            cv2.destroyAllWindows()
                            return

            except Exception as e:
                print(f"Errore durante il video {video_name}: {e}")
                traceback.print_exc()  # Stampa l'errore completo per debug

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