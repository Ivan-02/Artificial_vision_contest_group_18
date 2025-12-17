import os
import sys
import time
import yaml
import cv2
import traceback
import random
from ultralytics import YOLO
from tqdm import tqdm
from .common.io_manager import ReportManager
from .common.vis_utils import Visualizer
from .evaluator import Evaluator


class Tracker:
    def __init__(self, config, conf_mode):
        self.cfg = config
        self.cfg_mode = conf_mode
        self._check_keys()

        print("Caricamento modello...")
        self.model = YOLO(self.cfg['paths']['model_weights'])

        # --- 1. Setup Output Manager (Gestisce cartelle e JSON) ---
        base_out = os.path.join(self.cfg['paths']['output_submission'], self.cfg_mode['test_name'], "track")
        self.reporter = ReportManager(base_out)

        # --- 2. Setup Visualizer (Gestisce OpenCV) ---
        # Viene istanziato solo se il display è abilitato
        self.enable_display = self.cfg_mode['display']
        if self.enable_display:
            self.vis = Visualizer(display_width=self.cfg_mode['display_width'])
        else:
            self.vis = None

        # --- 3. Salvataggio Configurazione Iniziale nel JSON ---
        self.reporter.update_json_section("configuration", self.cfg_mode)

        # Configurazione parametri Tracker
        self.conf = self.cfg_mode['conf_threshold']
        self.iou = self.cfg_mode['iou_threshold']
        self.max_video = self.cfg_mode.get('max_video', None)
        self.random_mode = self.cfg_mode['random']
        self.run_hota = self.cfg_mode['eval']

        # Generazione file YAML temporaneo per il tracker (specifico di Ultralytics)
        self.tracker_yaml_path = self.cfg['paths'].get('temp_tracker_yaml', './temp_tracker.yaml')
        os.makedirs(os.path.dirname(self.tracker_yaml_path), exist_ok=True)
        with open(self.tracker_yaml_path, 'w') as f:
            yaml.dump(self.cfg_mode['tracker_settings'], f, sort_keys=False)
        print(f"File configurazione tracker generato in: {self.tracker_yaml_path}")

    def _check_keys(self):
        """Validazione chiavi di configurazione (Invariato)"""
        required_keys = {
            'test_name', 'conf_threshold', 'iou_threshold', 'display', 'display_width',
            'half', 'show', 'imgsz', 'stream', 'verbose', 'persist',
            'agnostic_nms', 'classes', 'tracker_settings', 'eval', 'random'
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

    def get_video_list(self):
        """Logica di selezione video (Invariato)"""
        test_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['paths']['split'])
        if not os.path.exists(test_dir):
            print(f"ERRORE CRITICO: La cartella dati non esiste: {test_dir}")
            return []

        all_video_folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])

        if not all_video_folders:
            print(f"Nessuna cartella video trovata in {test_dir}")
            return []

        if self.max_video is not None and isinstance(self.max_video, int):
            if self.max_video < len(all_video_folders):
                if self.random_mode:
                    video_folders = random.sample(all_video_folders, self.max_video)
                    print(f"RANDOM MODE: Selezionati {len(video_folders)} video su {len(all_video_folders)} totali.")
                else:
                    video_folders = all_video_folders[:self.max_video]
                    print(
                        f"SEQUENTIAL MODE: Selezionati i primi {len(video_folders)} video su {len(all_video_folders)} totali.")
                return video_folders
            else:
                return all_video_folders
        else:
            return all_video_folders

    def track_video_generator(self, video_path_folder):
        """Generatore di frame e detection (Invariato - Core Logic)"""
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
            output_filename = f"tracking_{video_id}_{self.cfg['names']['team']}.txt"

            # Check esistenza file tramite il percorso gestito dal reporter
            if os.path.exists(os.path.join(self.reporter.output_dir, output_filename)):
                print(f"Skipping {video_name}, file esistente.")
                continue

            window_name = f"Monitor - {video_name}"
            start_time = time.time()

            # Buffer per accumulare le linee di un frame prima di scrivere
            frame_lines = []

            try:
                for frame_id, orig_img, detections in self.track_video_generator(video_name):

                    display_img = orig_img.copy() if self.enable_display else None
                    frame_lines.clear()  # Pulisce buffer frame precedente

                    for det in detections:
                        # 1. Preparazione dati per TXT
                        x_c, y_c, w, h = det['xywh']
                        x1 = x_c - (w / 2)
                        y1 = y_c - (h / 2)

                        line = f"{frame_id},{det['track_id']},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{det['conf']:.2f},-1,-1,-1\n"
                        frame_lines.append(line)

                        # 2. Preparazione grafica (Delegata al Visualizer)
                        if self.vis and display_img is not None:
                            self.vis.draw_box(
                                img=display_img,
                                box_xyxy=det['xyxy'],
                                track_id=det['track_id'],
                                conf=det['conf']
                            )

                    # 3. Scrittura su file (Delegata al Reporter - Append Mode)
                    if frame_lines:
                        self.reporter.save_txt_results(output_filename, frame_lines, append=True)

                    # 4. Visualizzazione (Delegata al Visualizer)
                    if self.vis and display_img is not None:
                        if not self.vis.show_frame(window_name, display_img):
                            print("\nInterruzione richiesta dall'utente.")
                            stop_requested = True
                            break

                # --- Fine Video ---
                elapsed_time = round(time.time() - start_time, 4)

                # Aggiornamento JSON Report (Delegato al Reporter)
                self.reporter.update_json_section("video_execution_times", {video_name: elapsed_time})

            except Exception as e:
                print(f"Errore durante il video {video_name}: {e}")
                traceback.print_exc()

            finally:
                # Chiusura finestra specifica video se aperta
                if self.vis:
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass

        # Chiusura globale finestre
        if self.vis:
            self.vis.close_windows()

        print(f"\nReport esecuzione salvato in: {self.reporter.json_path}")
        print("\n--- Tracking Completato ---")

        if self.run_hota:
            print("\n--- Avvio Valutazione Automatica (HOTA) ---")
            original_subdirs = self.cfg['paths'].get('output_subdirs', [])
            current_subdir = os.path.join(self.cfg_mode['test_name'], "track")
            self.cfg['paths']['output_subdirs'] = [current_subdir]

            evaluator = Evaluator(self.cfg)
            evaluator.run_hota()

            self.cfg['paths']['output_subdirs'] = original_subdirs