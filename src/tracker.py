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
from .field_detector import FieldFilter


class Tracker:
    def __init__(self, config, conf_mode):
        self.cfg = config
        self.cfg_mode = conf_mode
        self._check_keys()

        print(f"Caricamento modello YOLO: {self.cfg['paths']['model_weights']} ...")
        self.model = YOLO(self.cfg['paths']['model_weights'])

        base_out = os.path.join(self.cfg['paths']['output_submission'], self.cfg_mode['test_name'], "track")
        self.reporter = ReportManager(base_out)

        self.enable_display = self.cfg_mode['display'] or self.cfg_mode.get('show', False)

        if self.enable_display:
            self.vis = Visualizer(display_width=self.cfg_mode['display_width'])
        else:
            self.vis = None

        if FieldFilter:
            self.field_filter = FieldFilter(debug=self.enable_display)
        else:
            self.field_filter = None

        self.reporter.update_json_section("configuration", self.cfg_mode)

        self.conf = self.cfg_mode['conf_threshold']
        self.iou = self.cfg_mode['iou_threshold']
        self.max_video = self.cfg_mode.get('max_video', None)
        self.random_mode = self.cfg_mode['random']
        self.run_hota = self.cfg_mode['eval']

        self.specific_video = self.cfg_mode.get('specific_video', None)

        self.tracker_yaml_path = self.cfg['paths'].get('temp_tracker_yaml', './temp_tracker.yaml')
        os.makedirs(os.path.dirname(self.tracker_yaml_path), exist_ok=True)
        with open(self.tracker_yaml_path, 'w') as f:
            yaml.dump(self.cfg_mode['tracker_settings'], f, sort_keys=False)
        print(f"File configurazione tracker generato in: {self.tracker_yaml_path}")

    def _check_keys(self):

        required_keys = {
            'test_name', 'conf_threshold', 'iou_threshold', 'display', 'display_width',
            'half', 'show', 'imgsz', 'stream', 'verbose', 'persist',
            'agnostic_nms', 'classes', 'tracker_settings', 'eval', 'random', 'single_cls'
        }
        current_keys = set(self.cfg_mode.keys())
        missing_keys = required_keys - current_keys
        if missing_keys:
            print(f"[ERRORE CONFIG] Mancano le seguenti chiavi principali: {list(missing_keys)}")
            sys.exit(1)

    def get_video_list(self):
        test_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['paths']['split'])
        if not os.path.exists(test_dir):
            print(f"ERRORE CRITICO: Cartella dati non trovata: {test_dir}")
            return []

        all_video_folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])

        if not all_video_folders:
            return []

        if self.specific_video:
            target_str = str(self.specific_video)
            filtered_list = [v for v in all_video_folders if target_str in v]

            if filtered_list:
                selected = filtered_list[0]
                print(f"--- MODALITÀ SINGOLA: Analisi forzata del video '{selected}' ---")
                return [selected]
            else:
                print(f"ATTENZIONE: Il video specifico '{target_str}' non è stato trovato.")
                return []

        if self.max_video is not None and isinstance(self.max_video, int):
            if self.max_video < len(all_video_folders):
                if self.random_mode:
                    video_folders = random.sample(all_video_folders, self.max_video)
                else:
                    video_folders = all_video_folders[:self.max_video]
                return video_folders

        return all_video_folders

    def track_video_generator(self, video_path_folder):

        video_path_img = os.path.join(self.cfg['paths']['raw_data'],
                                      self.cfg['paths']['split'],
                                      video_path_folder, 'img1')

        if not os.path.exists(video_path_img):
            print(f"SKIP: Cartella immagini non trovata {video_path_folder}")
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
            show= self.cfg_mode['show'],
            agnostic_nms=self.cfg_mode['agnostic_nms'],
            imgsz=self.cfg_mode['imgsz'],
            half=self.cfg_mode['half'],
            single_cls=self.cfg_mode['single_cls']
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
            print("Visualizzazione attiva: Premi 'q' sulla finestra per interrompere.")

        video_folders = self.get_video_list()

        if not video_folders:
            print("Nessun video da analizzare.")
            return

        stop_requested = False

        for video_name in tqdm(video_folders, desc="Tracking Videos"):
            if stop_requested: break

            parts = video_name.split('-')
            video_id = parts[1].split('.')[0] if len(parts) > 1 else video_name

            team_id = self.cfg['names']['team']
            output_filename = f"tracking_{video_id}_{team_id}.txt"

            if os.path.exists(os.path.join(self.reporter.output_dir, output_filename)):
                print(f"Skipping {video_name}, file esistente.")
                continue

            window_name = f"Tracking - {video_name}"
            start_time = time.time()
            frame_lines = []

            try:
                for frame_id, orig_img, detections in self.track_video_generator(video_name):

                    if self.enable_display:
                        display_img = orig_img.copy()
                    else:
                        display_img = None

                    frame_lines.clear()

                    final_detections = detections
                    field_contour = None

                    if self.field_filter:
                        img_to_process = display_img if (display_img is not None) else orig_img

                        final_detections, field_contour = self.field_filter.filter_detections(img_to_process,
                                                                                              detections)

                    for det in final_detections:
                        x_c, y_c, w, h = det['xywh']
                        x1 = x_c - (w / 2)
                        y1 = y_c - (h / 2)

                        line = f"{frame_id},{det['track_id']},{int(x1)},{int(y1)},{int(w)},{int(h)}\n"
                        frame_lines.append(line)

                        if self.vis and display_img is not None:
                            self.vis.draw_box(
                                img=display_img,
                                box_xyxy=det['xyxy'],
                                track_id=det['track_id'],
                                conf=det['conf']
                            )

                    if frame_lines:
                        self.reporter.save_txt_results(output_filename, frame_lines, append=True)

                    if self.vis and display_img is not None:
                        if self.field_filter and field_contour is not None:
                            self.field_filter.draw_debug(display_img, field_contour)

                        if not self.vis.show_frame(window_name, display_img):
                            print("\nInterruzione richiesta dall'utente (Q).")
                            stop_requested = True
                            break

                elapsed_time = round(time.time() - start_time, 4)
                self.reporter.update_json_section("video_execution_times", {video_name: elapsed_time})

            except Exception as e:
                print(f"Errore durante il tracking di {video_name}:")
                traceback.print_exc()

            finally:
                if self.vis:
                    try:
                        cv2.destroyWindow(window_name)
                    except:
                        pass

        if self.vis:
            self.vis.close_windows()

        print(f"\nTracking Completato. Risultati in: {self.reporter.output_dir}")

        if self.run_hota and not stop_requested:
            print("\n--- Avvio Valutazione Automatica (HOTA) ---")
            original_subdirs = self.cfg['paths'].get('output_subdirs', [])
            current_subdir = os.path.join(self.cfg_mode['test_name'], "track")
            self.cfg['paths']['output_subdirs'] = [current_subdir]

            evaluator = Evaluator(self.cfg)
            evaluator.run_hota()

            self.cfg['paths']['output_subdirs'] = original_subdirs