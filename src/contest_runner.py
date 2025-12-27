import os
from tqdm import tqdm
from .tracker import Tracker
from .common.io_manager import ReportManager
from .common.vis_utils import Visualizer
from .common.data_utils import GeometryUtils
from .field_detector import FieldFilter
from .behavior import BehaviorAnalyzer


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

        self.reporter_track = ReportManager(common_out_dir)
        self.reporter_behav = ReportManager(common_out_dir)

        print(f"[ContestRunner] Output Directory (Unificata): {common_out_dir}")

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
        """
        Carica le Region of Interest (ROI) specifiche per il video in esame, delegando l'operazione
        alla logica definita nell'analizzatore di comportamento.
        """
        return BehaviorAnalyzer._load_rois(self, video_folder)

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

            rois_data = self._load_rois(video_name)

            window_name = f"Contest - {video_name}"

            try:
                for frame_id, img, detections in self.tracker.track_video_generator(video_name):
                    h_img, w_img = img.shape[:2]

                    lines_track = []
                    lines_behav = []

                    final_detections = detections
                    if self.field_filter:
                        final_detections, _ = self.field_filter.filter_detections(img, detections)

                    for det in final_detections:
                        x_c, y_c, w, h = det['xywh']
                        x1 = int(x_c - (w / 2))
                        y1 = int(y_c - (h / 2))
                        lines_track.append(f"{frame_id},{det['track_id']},{x1},{y1},{int(w)},{int(h)}\n")

                    count_roi1 = 0
                    count_roi2 = 0

                    for det in final_detections:
                        if GeometryUtils.is_in_roi(det['xywh'], rois_data.get('roi1'), w_img, h_img):
                            count_roi1 += 1
                        if GeometryUtils.is_in_roi(det['xywh'], rois_data.get('roi2'), w_img, h_img):
                            count_roi2 += 1

                    lines_behav.append(f"{frame_id},1,{count_roi1}\n")
                    lines_behav.append(f"{frame_id},2,{count_roi2}\n")

                    if lines_track:
                        self.reporter_track.save_txt_results(fname_track, lines_track, append=True)
                    if lines_behav:
                        self.reporter_behav.save_txt_results(fname_behav, lines_behav, append=True)

                    if self.vis and self.enable_display:
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