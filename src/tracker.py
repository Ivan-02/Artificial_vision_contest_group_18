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
    """
    Classe principale che gestisce l'intero workflow di tracking degli oggetti nei video,
    integrando il rilevamento YOLO, il filtraggio spaziale del campo e la gestione dei report.
    """

    def __init__(self, config, conf_mode):
        """
        Inizializza la pipeline di tracking configurando il modello YOLO, il sistema di visualizzazione,
        i filtri per il campo da gioco e i parametri di salvataggio dei risultati.
        """
        self.cfg = config
        self.cfg_mode = conf_mode

        print(f"\n[Tracker] {'=' * 50}")
        print("[Tracker] INIZIALIZZAZIONE PIPELINE")
        print(f"[Tracker] {'=' * 50}")

        self._check_keys()

        weights_path = self.cfg['paths']['model_weights']
        print(f"[Tracker] [INIT] Caricamento modello YOLO...")
        print(f"[Tracker]        Path: {os.path.basename(weights_path)}")
        self.model = YOLO(weights_path)

        base_out = os.path.join(self.cfg['paths']['output_submission'], self.cfg_mode['test_name'], "track")
        self.reporter = ReportManager(base_out)
        print(f"[Tracker] [INIT] Output directory: {base_out}")

        self.enable_display = self.cfg_mode['display'] or self.cfg_mode.get('show', False)
        if self.enable_display:
            print(f"[Tracker] [INIT] Visualizzazione: ATTIVA (Width: {self.cfg_mode['display_width']}px)")
            self.vis = Visualizer(display_width=self.cfg_mode['display_width'])
        else:
            print(f"[Tracker] [INIT] Visualizzazione: DISATTIVA (Headless Mode)")
            self.vis = None

        if FieldFilter:
            field_settings = self.cfg_mode.get('field_det_settings', {})
            self.field_filter = FieldFilter(settings=field_settings)
            print(f"[Tracker] [INIT] FieldFilter: Caricato")
        else:
            self.field_filter = None
            print(f"[Tracker] [INIT] FieldFilter: Non disponibile")

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

    def _check_keys(self):
        """
        Esegue una validazione dei parametri di configurazione per garantire che tutte
        le chiavi obbligatorie siano presenti nel dizionario di input.
        """
        required_keys = {
            'test_name', 'conf_threshold', 'iou_threshold', 'display', 'display_width',
            'half', 'show', 'imgsz', 'stream', 'verbose', 'persist',
            'agnostic_nms', 'classes', 'tracker_settings', 'eval', 'random', 'single_cls'
        }
        current_keys = set(self.cfg_mode.keys())
        missing_keys = required_keys - current_keys
        if missing_keys:
            print(f"[Tracker] [ERROR] Configurazione incompleta! Mancano le chiavi: {list(missing_keys)}")
            sys.exit(1)

    def get_video_list(self):
        """
        Analizza la directory sorgente e restituisce l'elenco dei video da elaborare,
        applicando eventuali filtri (video specifici, campionamento casuale o limiti numerici).
        """
        test_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['paths']['split'])
        if not os.path.exists(test_dir):
            print(f"[Tracker] [ERROR] CRITICO: Cartella dati non trovata -> {test_dir}")
            return []

        all_video_folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])

        if not all_video_folders:
            return []

        if self.specific_video:
            target_str = str(self.specific_video)
            filtered_list = [v for v in all_video_folders if target_str in v]

            if filtered_list:
                selected = filtered_list[0]
                print(f"[Tracker] [!] MODALITÀ SINGOLA: Forzata analisi su '{selected}'")
                return [selected]
            else:
                print(f"[Tracker] [!] ATTENZIONE: Video specifico '{target_str}' non trovato.")
                return []

        if self.max_video is not None and isinstance(self.max_video, int):
            if self.max_video < len(all_video_folders):
                if self.random_mode:
                    video_folders = random.sample(all_video_folders, self.max_video)
                    print(f"[Tracker] [!] Random Sampling: Selezionati {self.max_video} video a caso.")
                else:
                    video_folders = all_video_folders[:self.max_video]
                    print(f"[Tracker] [!] Limit: Selezionati i primi {self.max_video} video.")
                return video_folders

        return all_video_folders

    def track_video_generator(self, video_path_folder):
        """
        Generatore che esegue il tracking YOLO sui frame di un video, restituendo iterativamente
        l'ID del frame, l'immagine originale e l'elenco delle rilevazioni effettuate.
        """
        video_path_img = os.path.join(self.cfg['paths']['raw_data'],self.cfg['paths']['split'],video_path_folder, 'img1')

        if not os.path.exists(video_path_img):
            tqdm.write(f"[Tracker] [!] SKIP: Cartella immagini assente in {video_path_folder}")
            return

        is_verbose = self.cfg_mode.get('verbose', False)

        results_generator = self.model.track(
            source=video_path_img,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker_yaml_path,
            classes=self.cfg_mode['classes'],
            persist=self.cfg_mode['persist'],
            device=self.cfg['device'],
            verbose=is_verbose,
            stream=self.cfg_mode['stream'],
            show=self.cfg_mode['show'],
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
        """
        Orchestra l'esecuzione del tracking su tutti i video selezionati, gestisce il filtraggio
        delle coordinate, il salvataggio dei file di output e l'eventuale valutazione finale (HOTA).
        """

        print(f"\n[Tracker] {'=' * 50}")
        print("[Tracker] AVVIO: ESECUZIONE SU DATASET")
        print(f"[Tracker] {'=' * 50}")
        if self.enable_display:
            print("[Tracker] [INFO] Premi 'q' sulla finestra video per interrompere l'analisi.")

        video_folders = self.get_video_list()

        if not video_folders:
            print("[Tracker] [!] Nessun video da analizzare. Uscita.")
            return

        stop_requested = False
        print(f"[Tracker] Video in coda: {len(video_folders)}")

        for video_name in tqdm(video_folders, desc="[Tracker] Progress", unit="vid"):
            if stop_requested:
                break

            parts = video_name.split('-')
            video_id = parts[1].split('.')[0] if len(parts) > 1 else video_name

            team_id = self.cfg['names']['team']
            output_filename = f"tracking_{video_id}_{team_id}.txt"
            output_path_full = os.path.join(self.reporter.output_dir, output_filename)

            if os.path.exists(output_path_full):
                continue

            window_name = f"Tracking - {video_name}"
            start_time = time.time()
            frame_lines = []
            current_video_interrupted = False

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
                        final_detections, field_contour = self.field_filter.filter_detections(orig_img, detections)

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
                        if self.field_filter:
                            display_img = self.field_filter.draw_debug(display_img, field_contour)

                        if not self.vis.show_frame(window_name, display_img):
                            tqdm.write(f"\n[Tracker] [STOP] Interruzione utente su video: {video_name}")
                            stop_requested = True
                            current_video_interrupted = True
                            break

                elapsed_time = round(time.time() - start_time, 4)
                self.reporter.update_json_section("video_execution_times", {video_name: elapsed_time})

            except Exception as e:
                tqdm.write(f"[Tracker] [ERROR] Errore su {video_name}: {e}")
                traceback.print_exc()

            finally:
                if self.vis:
                    try:
                        cv2.destroyWindow(window_name)
                    except:
                        pass

                if current_video_interrupted and os.path.exists(output_path_full):
                    try:
                        os.remove(output_path_full)
                        tqdm.write(f"[Tracker] [INFO] Cancellato file parziale: {output_filename}")
                    except OSError as e:
                        tqdm.write(f"[Tracker] [WARN] Impossibile cancellare file parziale: {e}")

        if self.vis:
            self.vis.close_windows()

        print(f"\n[Tracker] [✔] Tracking completato (o interrotto).")
        print(f"[Tracker]     Output salvato in: {os.path.abspath(self.reporter.output_dir)}")

        if self.run_hota:
            print("\n[Tracker] --- Avvio Valutazione Automatica (HOTA) sui video processati ---")
            original_subdirs = self.cfg['paths'].get('output_subdirs', [])
            current_subdir = os.path.join(self.cfg_mode['test_name'], "track")
            self.cfg['paths']['output_subdirs'] = [current_subdir]

            evaluator = Evaluator(self.cfg)
            evaluator.run_hota()
            self.cfg['paths']['output_subdirs'] = original_subdirs