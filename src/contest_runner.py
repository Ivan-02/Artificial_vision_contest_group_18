import os
import cv2
from tqdm import tqdm
from .tracker import Tracker
from .common.io_manager import ReportManager
from .common.vis_utils import Visualizer
from .common.data_utils import GeometryUtils
from .field_detector import FieldFilter

class ContestRunner:
    """
    Orchestra l'esecuzione simultanea della pipeline di tracking e dell'analisi del comportamento (Behavior),
    gestendo l'elaborazione dei video e la generazione coordinata dei file di output per la competizione.
    """
    def __init__(self, config, conf_mode):
        """
        Inizializza il runner configurando il tracker, i gestori dei report (ReportManager),
        i sistemi di visualizzazione e i filtri spaziali per il campo da gioco.
        """
        self.cfg = config
        self.conf_mode = conf_mode

        print(f"\n[ContestRunner] {'=' * 50}")
        print("[ContestRunner] AVVIO: MODALITÀ CONTEST (Track + Behavior)")
        print(f"[ContestRunner] {'=' * 50}")

        self.tracker = Tracker(config, conf_mode)

        common_out_dir = os.path.join(self.cfg['paths']['output_submission'], conf_mode['test_name'], "results")
        self.reporter = ReportManager(common_out_dir)

        print(f"[ContestRunner] Output Directory (Unificata): {common_out_dir}")

        self.enable_display = self.conf_mode.get('display', False)
        if self.enable_display:
            self.vis = Visualizer(display_width=conf_mode['display_width'])
        else:
            self.vis = None

        field_settings = self.conf_mode.get('field_det_settings')
        if field_settings:
            self.field_filter = FieldFilter(settings=field_settings)
        else:
            self.field_filter = None

        self.roi_colors = {
            'roi1': (0, 255, 255),
            'roi2': (255, 0, 255)
        }

    def run(self):
        """
        Esegue il ciclo principale di analisi su tutti i video: processa i frame, applica filtri spaziali,
        esegue il conteggio degli oggetti nelle ROI e salva i risultati di tracking e behavior in formato testo.
        """
        video_folders = self.tracker.get_video_list()

        for video_name in tqdm(video_folders, desc="[Contest] Progress", unit="vid"):

            video_id = video_name
            team_id = self.cfg['names']['team']
            fname_track = f"tracking_{video_id}_{team_id}.txt"
            fname_behav = f"behavior_{video_id}_{team_id}.txt"

            video_path = os.path.join(self.cfg["paths"]["raw_data"], self.cfg["paths"]["split"], video_name)
            roi_path = self.cfg["paths"]["roi"]
            rois_data = self.reporter.load_rois(video_path, roi_path)

            window_name = f"Contest - {video_name}"

            try:
                for frame_id, img, detections in self.tracker.track_video_generator(video_name):
                    h_img, w_img = img.shape[:2]

                    display_img = img.copy() if (self.vis and self.enable_display) else None

                    final_detections = detections
                    field_contour = None

                    if self.field_filter:
                        final_detections, field_contour = self.field_filter.filter_detections(img, detections)

                    count_roi1 = 0
                    count_roi2 = 0
                    lines_track = []

                    for det in final_detections:
                        x_c, y_c, w, h = det['xywh']
                        x1 = int(x_c - (w / 2))
                        y1 = int(y_c - (h / 2))
                        lines_track.append(f"{frame_id},{det['track_id']},{x1},{y1},{int(w)},{int(h)}\n")

                        in_roi = False
                        color = (0, 0, 255)

                        if GeometryUtils.is_in_roi(det['xywh'], rois_data.get('roi1'), w_img, h_img):
                            count_roi1 += 1
                            in_roi = True
                            color = (0, 255, 255)

                        if GeometryUtils.is_in_roi(det['xywh'], rois_data.get('roi2'), w_img, h_img):
                            count_roi2 += 1
                            if not in_roi: color = (255, 0, 255)

                        if display_img is not None and self.vis:
                            self.vis.draw_box(display_img, det['xyxy'], det['track_id'], det['conf'], color=color)

                    lines_behav = [
                        f"{frame_id},1,{count_roi1}\n",
                        f"{frame_id},2,{count_roi2}\n"
                    ]

                    if lines_track:
                        self.reporter.save_txt_results(fname_track, lines_track, append=True)
                    if lines_behav:
                        self.reporter.save_txt_results(fname_behav, lines_behav, append=True)

                    if display_img is not None and self.vis:

                        current_counts = {'roi1': count_roi1, 'roi2': count_roi2}

                        for key, roi_def in rois_data.items():
                            val = current_counts.get(key, 0)
                            col = self.roi_colors.get(key, (255, 255, 255))
                            self.vis.draw_roi(display_img, roi_def, val, color=col, label_key=key.upper())

                        if self.field_filter:
                            display_img = self.field_filter.draw_debug(display_img, field_contour)

                        if not self.vis.show_frame(window_name, display_img):
                            tqdm.write(f"[Contest] Interruzione utente su video {video_name}")
                            break

            except Exception as e:
                tqdm.write(f"[ERROR] Errore critico su video {video_name}: {e}")
            finally:
                if self.vis:
                    try:
                        cv2.destroyWindow(window_name)
                    except:
                        pass

        if self.vis:
            self.vis.close_windows()