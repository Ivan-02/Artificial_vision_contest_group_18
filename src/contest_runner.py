import os
from tqdm import tqdm
from .tracker import Tracker
from .common.io_manager import ReportManager
from .common.vis_utils import Visualizer
from .common.data_utils import GeometryUtils
from .field_detector import FieldFilter
from .behavior import BehaviorAnalyzer


class ContestRunner:
    def __init__(self, config, conf_mode):
        self.cfg = config
        self.conf_mode = conf_mode

        print(f"\n[ContestRunner] {'=' * 50}")
        print("[ContestRunner] AVVIO: MODALITÀ CONTEST (Track + Behavior)")
        print(f"[ContestRunner] {'=' * 50}")

        # Inizializza il tracker (gestisce il modello YOLO e il caricamento video)
        self.tracker = Tracker(config, conf_mode)

        # --- Setup Output Managers ---
        # 1. Output per il Tracking
        track_out_dir = os.path.join(self.cfg['paths']['output_submission'], conf_mode['test_name'], "track")
        self.reporter_track = ReportManager(track_out_dir)

        # 2. Output per la Behavior Analysis
        behav_out_dir = os.path.join(self.cfg['paths']['output_submission'], conf_mode['test_name'], "behavior")
        self.reporter_behav = ReportManager(behav_out_dir)

        print(f"[ContestRunner] Output Tracking: {track_out_dir}")
        print(f"[ContestRunner] Output Behavior: {behav_out_dir}")

        # --- Setup Visualizer e Filtri ---
        self.enable_display = self.conf_mode.get('display', False)
        if self.enable_display:
            self.vis = Visualizer(display_width=conf_mode['display_width'])
        else:
            self.vis = None

        if FieldFilter:
            field_settings = self.conf_mode.get('field_det_settings', {})
            self.field_filter = FieldFilter(settings=field_settings)
        else:
            self.field_filter = None

    def _load_rois(self, video_folder):
        return BehaviorAnalyzer._load_rois(self, video_folder)

    def run(self):
        video_folders = self.tracker.get_video_list()

        # Se i video sono già rinominati (es. "01", "02"), video_name è direttamente l'ID
        for video_name in tqdm(video_folders, desc="[Contest] Progress", unit="vid"):

            # Assegnazione diretta (rimossa la logica split '-')
            video_id = video_name

            team_id = self.cfg['names']['team']

            # File di output
            fname_track = f"tracking_{video_id}_{team_id}.txt"
            fname_behav = f"behavior_{video_id}_{team_id}.txt"

            # Carica ROI specifiche per il video
            rois_data = self._load_rois(video_name)

            window_name = f"Contest - {video_name}"

            try:
                # --- CICLO UNICO SUI FRAME ---
                for frame_id, img, detections in self.tracker.track_video_generator(video_name):
                    h_img, w_img = img.shape[:2]

                    lines_track = []
                    lines_behav = []

                    # 1. Filtraggio Campo (Opzionale)
                    final_detections = detections
                    if self.field_filter:
                        final_detections, _ = self.field_filter.filter_detections(img, detections)

                    # --- LOGICA TRACKING ---
                    for det in final_detections:
                        x_c, y_c, w, h = det['xywh']
                        x1 = int(x_c - (w / 2))
                        y1 = int(y_c - (h / 2))
                        # Nota: Salviamo solo int per risparmiare spazio e rispettare il formato
                        lines_track.append(f"{frame_id},{det['track_id']},{x1},{y1},{int(w)},{int(h)}\n")

                    # --- LOGICA BEHAVIOR ---
                    count_roi1 = 0
                    count_roi2 = 0

                    for det in final_detections:
                        # Conta se il centro del box (o i piedi) sono nella ROI
                        if GeometryUtils.is_in_roi(det['xywh'], rois_data.get('roi1'), w_img, h_img):
                            count_roi1 += 1
                        if GeometryUtils.is_in_roi(det['xywh'], rois_data.get('roi2'), w_img, h_img):
                            count_roi2 += 1

                    lines_behav.append(f"{frame_id},1,{count_roi1}\n")
                    lines_behav.append(f"{frame_id},2,{count_roi2}\n")

                    # 2. SALVATAGGIO SU DISCO (Entrambi i file, modalità append)
                    if lines_track:
                        self.reporter_track.save_txt_results(fname_track, lines_track, append=True)
                    if lines_behav:
                        self.reporter_behav.save_txt_results(fname_behav, lines_behav, append=True)

                    # 3. VISUALIZZAZIONE (Opzionale)
                    if self.vis and self.enable_display:
                        # Qui potresti chiamare self.vis.draw_box e self.vis.draw_roi se vuoi vedere il video
                        if not self.vis.show_frame(window_name, img):
                            print(f"[Contest] Interruzione utente su video {video_name}")
                            break

            except Exception as e:
                print(f"[ERROR] Errore critico su video {video_name}: {e}")
            finally:
                if self.vis:
                    try:
                        import cv2
                        cv2.destroyWindow(window_name)
                    except:
                        pass

        if self.vis:
            self.vis.close_windows()