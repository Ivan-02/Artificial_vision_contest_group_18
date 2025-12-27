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
    """
        Gestisce la pipeline di behavior analisys, coordinando il tracking degli oggetti,
        il filtraggio basato sul campo di gioco e il conteggio delle presenze all'interno
        delle Region of Interest (ROI) configurate, producendo report e visualizzazioni.
        """

    def __init__(self, config, conf_mode):
        """
        Inizializza le componenti principali (tracker, visualizer, report manager) e imposta
        i percorsi di output e le configurazioni per il filtraggio del campo.
        """
        self.cfg = config
        self.conf_mode = conf_mode

        print(f"\n[BehaviorAnalyzer] {'=' * 50}")
        print("[BehaviorAnalyzer] INIZIALIZZAZIONE PIPELINE")
        print(f"[BehaviorAnalyzer] {'=' * 50}")

        self.tracker = Tracker(config, conf_mode)

        base_out = os.path.join(self.cfg['paths']['output_submission'], conf_mode['test_name'], "behavior")
        self.reporter = ReportManager(base_out)
        print(f"[BehaviorAnalyzer] [INIT] Output directory: {base_out}")

        self.enable_display = self.conf_mode.get('display', False)
        if self.enable_display:
            print(f"[BehaviorAnalyzer] [INIT] Visualizzazione: ATTIVA (Width: {self.conf_mode['display_width']}px)")
            self.vis = Visualizer(display_width=conf_mode['display_width'])
        else:
            print(f"[BehaviorAnalyzer] [INIT] Visualizzazione: DISATTIVA")
            self.vis = None

        if FieldFilter:
            field_settings = self.conf_mode.get('field_det_settings', {})
            self.field_filter = FieldFilter(settings=field_settings)
            print(f"[BehaviorAnalyzer] [INIT] FieldFilter: Caricato")
        else:
            self.field_filter = None
            print(f"[BehaviorAnalyzer] [INIT] FieldFilter: Non disponibile")

        self.run_eval = conf_mode.get('eval', False)

        self.colors = {
            'default': (0, 0, 255),  # Rosso
            'roi1': (0, 255, 255),  # Giallo
            'roi2': (255, 0, 255)  # Magenta
        }

        self.reporter.update_json_section("configuration", self.conf_mode)

    def _load_rois(self, video_folder):
        """
        Carica i dati delle ROI da file JSON esterno; in caso di errore o file mancante,
        restituisce un set di coordinate di fallback predefinito.
        """

        json_path = self.cfg['paths']['roi']
        fallback = {
            "roi1": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.4},
            "roi2": {"x": 0.5, "y": 0.7, "width": 0.5, "height": 0.3}
        }
        if not os.path.exists(json_path):
            return fallback
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            tqdm.write(f"[BehaviorAnalyzer] [WARN] Errore caricamento ROI: {e}. Uso fallback.")
            return fallback

    def run(self):
        """
        Avvia l'elaborazione sequenziale dei video: gestisce il ciclo frame-by-frame,
        applica la logica di conteggio nelle ROI, aggiorna l'interfaccia grafica (se attiva)
        e salva i risultati e le metriche di tempo su file.
        """
        print(f"\n[BehaviorAnalyzer] {'=' * 50}")
        print("[BehaviorAnalyzer] AVVIO: BEHAVIOR ANALYSIS")
        print(f"[BehaviorAnalyzer] {'=' * 50}")
        if self.enable_display:
            print("[BehaviorAnalyzer] [INFO] Premi 'q' sulla finestra video per interrompere l'analisi.")

        video_folders = self.tracker.get_video_list()

        if not video_folders:
            print("[BehaviorAnalyzer] [!] Nessun video da analizzare. Uscita.")
            return

        stop_execution = False
        print(f"[BehaviorAnalyzer] Video in coda: {len(video_folders)}")

        for video_name in tqdm(video_folders, desc="[BehaviorAnalyzer] Progress", unit="vid"):
            if stop_execution: break

            parts = video_name.split('-')
            video_id = parts[1].split('.')[0] if len(parts) > 1 else video_name

            team_id = self.cfg['names']['team']
            output_filename = f"behavior_{video_id}_{team_id}.txt"
            output_path_full = os.path.join(self.reporter.output_dir, output_filename)

            if os.path.exists(output_path_full):
                continue

            window_name = f"Behavior - {video_name}"
            rois_data = self._load_rois(video_name)

            start_time = time.time()
            frame_lines = []
            current_video_interrupted = False

            try:
                for frame_id, img, detections in self.tracker.track_video_generator(video_name):
                    h_img, w_img = img.shape[:2]
                    frame_lines.clear()

                    display_img = img.copy() if self.enable_display else None

                    final_detections = detections
                    field_contour = None

                    if self.field_filter:
                        final_detections, field_contour = self.field_filter.filter_detections(img, detections)

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
                            if not in_roi:
                                assigned_color = self.colors['roi2']

                        if self.vis and display_img is not None:
                            to_draw.append((det, assigned_color))

                    frame_lines.append(f"{frame_id},1,{count_roi1}\n")
                    frame_lines.append(f"{frame_id},2,{count_roi2}\n")

                    self.reporter.save_txt_results(output_filename, frame_lines, append=True)

                    if self.vis and display_img is not None:
                        current_counts = {'roi1': count_roi1, 'roi2': count_roi2}
                        for key, roi_def in rois_data.items():
                            c_val = current_counts.get(key, 0)
                            c_col = self.colors.get(key, (255, 255, 255))
                            self.vis.draw_roi(display_img, roi_def, c_val, color=c_col, label_key=key.upper())

                        for det, color in to_draw:
                            self.vis.draw_box(
                                img=display_img,
                                box_xyxy=det['xyxy'],
                                track_id=det['track_id'],
                                conf=det['conf'],
                                color=color
                            )

                        if self.field_filter:
                             display_img = self.field_filter.draw_debug(display_img, field_contour)

                        if not self.vis.show_frame(window_name, display_img):
                            tqdm.write(f"\n[BehaviorAnalyzer] [STOP] Interruzione utente su video: {video_name}")
                            stop_execution = True
                            current_video_interrupted = True
                            break

                elapsed_time = round(time.time() - start_time, 4)
                self.reporter.update_json_section("video_execution_times", {video_name: elapsed_time})

            except Exception as e:
                tqdm.write(f"[BehaviorAnalyzer] [ERROR] Errore su {video_name}: {e}")

            finally:
                if self.vis:
                    try:
                        cv2.destroyWindow(window_name)
                    except:
                        pass

                if current_video_interrupted and os.path.exists(output_path_full):
                    try:
                        os.remove(output_path_full)
                        tqdm.write(f"[BehaviorAnalyzer] [INFO] Cancellato file parziale: {output_filename}")
                    except OSError as e:
                        tqdm.write(f"[BehaviorAnalyzer] [WARN] Impossibile cancellare file parziale: {e}")

        if self.vis:
            self.vis.close_windows()

        print(f"\n[BehaviorAnalyzer] [✔] Analisi completata (o interrotta).")
        print(f"[BehaviorAnalyzer]     Output salvato in: {os.path.abspath(self.reporter.output_dir)}")

        if self.run_eval:
            print("\n[BehaviorAnalyzer] --- Avvio Valutazione Automatica (nMAE) sui video processati ---")
            original_subdirs = self.cfg['paths'].get('output_subdirs', [])
            current_subdir = os.path.join(self.conf_mode['test_name'], "behavior")
            self.cfg['paths']['output_subdirs'] = [current_subdir]

            evaluator = Evaluator(self.cfg)
            evaluator.run_behavior()

            self.cfg['paths']['output_subdirs'] = original_subdirs