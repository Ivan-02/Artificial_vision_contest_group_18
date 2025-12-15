import os
import sys
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import traceback
import random

from .evaluator import HotaEvaluator


class Tracker:
    def __init__(self, config, conf_mode):
        self.cfg = config
        self.cfg_mode = conf_mode
        self._check_keys()

        print("Caricamento modello...")
        self.model = YOLO(self.cfg['paths']['model_weights'])

        # Setup output directory per il tracking puro
        self.output_dir = os.path.join(self.cfg['paths']['output_submission'], self.cfg_mode['test_name'], "_track")
        os.makedirs(self.output_dir, exist_ok=True)

        self.conf = self.cfg_mode['conf_threshold']
        self.iou = self.cfg_mode['iou_threshold']

        self.tracker_yaml_path = self.cfg['paths'].get('temp_tracker_yaml', './temp_tracker.yaml')
        os.makedirs(os.path.dirname(self.tracker_yaml_path), exist_ok=True)

        self.display_width = self.cfg_mode['display_width']
        self.enable_display = self.cfg_mode['display']
        self.max_video = self.cfg_mode.get('max_video', None)
        self.run_hota = self.cfg_mode['hota']

        with open(self.tracker_yaml_path, 'w') as f:
            yaml.dump(self.cfg_mode['tracker_settings'], f, sort_keys=False)

        print(f"File configurazione tracker generato in: {self.tracker_yaml_path}")


    def _check_keys(self):
        required_keys = {
            'test_name', 'conf_threshold', 'iou_threshold', 'display', 'display_width',
            'half', 'show', 'imgsz', 'stream', 'verbose', 'persist',
            'agnostic_nms', 'classes', 'tracker_settings', 'hota'
        }

        required_tracker_keys = {
            'tracker_type', 'track_high_thresh', 'track_low_thresh',
            'new_track_thresh', 'track_buffer', 'match_thresh',
            'gmc_method', 'proximity_thresh', 'appearance_thresh',
            'with_reid', 'fuse_score', 'reid_model', 'model'
        }

        current_keys = set(self.cfg_mode.keys())

        if 'max_video' not in current_keys: current_keys.add('max_video')

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
        return tuple(np.random.randint(0, 255, size=3).tolist())

    def get_video_list(self):
        test_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['paths']['split'])
        if not os.path.exists(test_dir):
            print(f"ERRORE CRITICO: La cartella dati non esiste: {test_dir}") # Ripristinato messaggio originale
            return []

        all_video_folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

        if not all_video_folders:
            print(f"Nessuna cartella video trovata in {test_dir}")
            return []

        if self.max_video is not None and isinstance(self.max_video, int):
            if self.max_video < len(all_video_folders):
                video_folders = random.sample(all_video_folders, self.max_video)
                print(f"RANDOM MODE: Selezionati {len(video_folders)} video su {len(all_video_folders)} totali.")
                return video_folders
            else:
                print(f"Richiesti {self.max_video} video, ma ne esistono solo {len(all_video_folders)}. Presi tutti.")
                return all_video_folders
        else:
            print(f"Elaborazione di tutti i {len(all_video_folders)} video trovati.")
            return all_video_folders

    def track_video_generator(self, video_path_folder):

        video_path_img = os.path.join(self.cfg['paths']['raw_data'],
                                      self.cfg['paths']['split'],
                                      video_path_folder, 'img1')

        if not os.path.exists(video_path_img):
            print(f"SKIP: Percorso immagini non trovato per {video_path_folder}")
            return

        results_generator = self.model.track(
            source=video_path_img,
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

        for result in results_generator:
            if not hasattr(result, 'path'): continue

            try:
                frame_id = int(os.path.basename(result.path).split('.')[0])
            except ValueError:
                frame_id = 0

            detections = []
            if result.boxes.id is not None:
                boxes_xywh = result.boxes.xywh.cpu().numpy()
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.int().cpu().numpy()



                for i, track_id in enumerate(track_ids):

                    current_class = cls_ids[i]

                    if current_class == 0:
                        continue

                    detections.append({
                        'track_id': track_id,
                        'cls_id': cls_ids[i],
                        'conf': confs[i],
                        'xywh': boxes_xywh[i],
                        'xyxy': boxes_xyxy[i]
                    })

            yield frame_id, result.orig_img, detections

    def run(self):
        print(f"--- Inizio Tracking ---")
        if self.enable_display:
            print("Visualizzazione attiva: Premi 'q' sulla finestra video per interrompere.")

        video_folders = self.get_video_list()

        stop_requested = False

        for video_name in tqdm(video_folders, desc="Tracking Videos"):

            if stop_requested: break

            video_id = video_name.split('-')[1].split('.')[0]
            output_txt = os.path.join(self.output_dir, f"tracking_{video_id}_{self.cfg['names']['team']}.txt")

            if os.path.exists(output_txt):
                print(f"Skipping {video_name}, file esistente.")
                continue

            window_name = f"Monitor - {video_name}"
            window_created = False

            f_out = open(output_txt, 'w')

            try:
                for frame_id, orig_img, detections in self.track_video_generator(video_name):

                    display_img = orig_img.copy() if self.enable_display else None

                    for det in detections:
                        x_c, y_c, w, h = det['xywh']
                        x1 = x_c - (w / 2)
                        y1 = y_c - (h / 2)

                        line = f"{frame_id},{det['track_id']},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{det['conf']:.2f},-1,-1,-1\n"
                        f_out.write(line)

                        if self.enable_display:
                            self._draw_on_frame(display_img, det)

                    f_out.flush()

                    if self.enable_display:
                        if not window_created and display_img is not None:
                            window_created = True

                        if not self._show_frame(window_name, display_img):
                            print("\nInterruzione richiesta dall'utente.")
                            stop_requested = True
                            break

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

        if self.run_hota:
            print("\n--- Avvio Valutazione Automatica (HOTA) ---")
            original_subdirs = self.cfg['paths'].get('output_subdirs', [])
            current_subdir = os.path.join(self.cfg_mode['test_name'], "_track")
            self.cfg['paths']['output_subdirs'] = [current_subdir]
            evaluator = HotaEvaluator(self.cfg)
            evaluator.run()
            self.cfg['paths']['output_subdirs'] = original_subdirs

    def _draw_on_frame(self, img, det):
        color = self._get_id_color(det['track_id'])
        x1, y1, x2, y2 = map(int, det['xyxy'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{det['track_id']} ({det['conf']:.2f})"

        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w_text, y1),
                      color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _show_frame(self, window_name, img):
        h, w = img.shape[:2]
        if h > 0 and w > 0:
            aspect = h / w
            d_h = int(self.display_width * aspect)
            resized = cv2.resize(img, (self.display_width, d_h))
            cv2.imshow(window_name, resized)
            if cv2.waitKey(1) & 0xFF == ord('q'): return False
        return True