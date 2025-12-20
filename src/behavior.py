import os
import time
import json
import cv2
from tqdm import tqdm
from .tracker import Tracker
from .evaluator import Evaluator
from .common.io_manager import ReportManager
from .common.vis_utils import Visualizer
from .common.data_utils import GeometryUtils
from .field_detector import FieldFilter


class BehaviorAnalyzer:
    def __init__(self, config, conf_mode):
        self.cfg = config
        self.conf_mode = conf_mode

        # Inizializza il tracker (che gestisce modello e inferenza)
        self.tracker = Tracker(config, conf_mode)

        # --- 1. Setup Output Manager ---
        base_out = os.path.join(self.cfg['paths']['output_submission'], conf_mode['test_name'], "behavior")
        self.reporter = ReportManager(base_out)

        # --- 2. Setup Visualizer ---
        self.enable_display = self.conf_mode.get('display', False)
        if self.enable_display:
            self.vis = Visualizer(display_width=conf_mode['display_width'])
        else:
            self.vis = None

        if FieldFilter:
            self.field_filter = FieldFilter(debug=self.enable_display)
        else:
            self.field_filter = None

        self.run_eval = conf_mode.get('eval', False)

        # Mappa colori specifica per la logica Behavior
        self.colors = {
            'default': (0, 0, 255),  # Rosso
            'roi1': (0, 255, 255),  # Giallo
            'roi2': (255, 0, 255)  # Magenta
        }

        # Salvataggio configurazione iniziale
        self.reporter.update_json_section("configuration", self.conf_mode)

    def _load_rois(self, video_folder):
        """Carica le ROI specifiche per il video (o usa fallback)."""
        # Nota: Qui mantengo la logica specifica del Behavior perché è business logic,
        # non I/O generico.
        json_path = self.cfg['paths']['roi']

        if not os.path.exists(json_path):
            return {"roi1": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.4},
                    "roi2": {"x": 0.5, "y": 0.7, "width": 0.5, "height": 0.3}}

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Errore caricamento ROI: {e}. Uso fallback.")
            return {"roi1": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.4},
                    "roi2": {"x": 0.5, "y": 0.7, "width": 0.5, "height": 0.3}}

    def run(self):
        video_folders = self.tracker.get_video_list()
        stop_execution = False

        for video_name in tqdm(video_folders, desc="Behavior Analysis"):
            if stop_execution: break

            output_filename = f"behavior_{video_name}_{self.cfg['names']['team']}.txt"

            # Check file esistente tramite reporter
            if os.path.exists(os.path.join(self.reporter.output_dir, output_filename)):
                print(f"\nSkipping {video_name}, file esistente.")
                continue

            window_name = f"Behavior - {video_name}"
            rois_data = self._load_rois(video_name)

            start_time = time.time()
            frame_lines = []

            try:
                for frame_id, img, detections in self.tracker.track_video_generator(video_name):
                    h_img, w_img = img.shape[:2]
                    frame_lines.clear()

                    display_img = img.copy() if self.enable_display else None

                    final_detections = detections
                    field_contour = None

                    if self.field_filter:
                        img_to_process = display_img if (display_img is not None) else img
                        final_detections, field_contour = self.field_filter.filter_detections(img_to_process,
                                                                                              detections)
                    count_roi1 = 0
                    count_roi2 = 0

                    to_draw = []

                    for det in final_detections:
                        assigned_color = self.colors['default']
                        in_roi = False

                        if GeometryUtils.is_in_roi(det['xywh'], rois_data.get('roi1'), w_img, h_img):
                            count_roi1 += 1
                            assigned_color = self.colors['roi1']
                            in_roi = True

                        if GeometryUtils.is_in_roi(det['xywh'], rois_data.get('roi2'), w_img, h_img):
                            count_roi2 += 1
                            if not in_roi:  # Se non era già nella 1, coloralo come 2
                                assigned_color = self.colors['roi2']

                        if self.vis and display_img is not None:
                            to_draw.append((det, assigned_color))

                    frame_lines.append(f"{frame_id},1,{count_roi1}\n")
                    frame_lines.append(f"{frame_id},2,{count_roi2}\n")

                    self.reporter.save_txt_results(output_filename, frame_lines, append=True)

                    # Visualizzazione
                    if self.vis and display_img is not None:
                        # 1. Disegna Contorno Campo (Debug)
                        if self.field_filter and field_contour is not None:
                            self.field_filter.draw_debug(display_img, field_contour)

                        # 2. Disegna ROI
                        current_counts = {'roi1': count_roi1, 'roi2': count_roi2}
                        for key, roi_def in rois_data.items():
                            c_val = current_counts.get(key, 0)
                            c_col = self.colors.get(key, (255, 255, 255))
                            self.vis.draw_roi(display_img, roi_def, c_val, color=c_col, label_key=key.upper())

                        # 3. Disegna Giocatori
                        for det, color in to_draw:
                            self.vis.draw_box(
                                img=display_img,
                                box_xyxy=det['xyxy'],
                                track_id=det['track_id'],
                                conf=det['conf'],
                                color=color
                            )

                        if not self.vis.show_frame(window_name, display_img):
                            print("\nInterruzione richiesta dall'utente (Tasto Q).")
                            stop_execution = True
                            break

                elapsed_time = round(time.time() - start_time, 4)
                self.reporter.update_json_section("video_execution_times", {video_name: elapsed_time})

            except Exception as e:
                print(f"Errore durante behavior analysis di {video_name}: {e}")

            finally:
                if self.vis:
                    try:
                        cv2.destroyWindow(window_name)
                    except:
                        pass

        if self.vis:
            self.vis.close_windows()

        print(f"Behavior Analysis Completata. File salvati in {self.reporter.output_dir}")

        if self.run_eval:
            print("\n--- Avvio Valutazione Automatica (nMAE) ---")
            original_subdirs = self.cfg['paths'].get('output_subdirs', [])
            current_subdir = os.path.join(self.conf_mode['test_name'], "behavior")
            self.cfg['paths']['output_subdirs'] = [current_subdir]

            evaluator = Evaluator(self.cfg)
            evaluator.run_behavior()

            self.cfg['paths']['output_subdirs'] = original_subdirs