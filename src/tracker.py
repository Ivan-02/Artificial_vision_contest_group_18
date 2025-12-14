import os
import cv2
import yaml  # Import necessario per scrivere il file
from ultralytics import YOLO
from tqdm import tqdm


class Tracker:
    def __init__(self, config):
        self.cfg = config
        self.model = YOLO(self.cfg['paths']['model_weights'])

        self.output_dir = self.cfg['paths']['output_submission']
        os.makedirs(self.output_dir, exist_ok=True)

        self.conf = self.cfg['tracking']['conf_threshold']
        self.iou = self.cfg['tracking']['iou_threshold']

        self.tracker_yaml_path = self.cfg['paths'].get('temp_tracker_yaml', './temp_tracker.yaml')

        tracker_settings = self.cfg['tracking']['tracker_settings']

        with open(self.tracker_yaml_path, 'w') as f:
            yaml.dump(tracker_settings, f, sort_keys=False)

        print(f"File configurazione tracker generato in: {self.tracker_yaml_path}")
        # ---------------------------------------------

    def run(self):
        print(f"--- Inizio Tracking ---")

        test_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['val']['split'])
        video_folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

        for video_name in tqdm(video_folders, desc="Tracking Videos"):
            video_path = os.path.join(test_dir, video_name, 'img1')
            output_txt = os.path.join(self.output_dir, f"{video_name}.txt")

            if os.path.exists(output_txt):
                continue

            results = self.model.track(
                source=video_path,
                conf=self.conf,
                iou=self.iou,

                tracker=self.tracker_yaml_path,

                classes=[1, 2, 3],  # Ignora palla
                persist=True,
                device=self.cfg['training']['device'],
                verbose=False,
                stream=True
            )

            with open(output_txt, 'w') as f:
                for result in results:
                    # Gestione frame ID se stream=True
                    if hasattr(result, 'path'):
                        frame_idx = os.path.basename(result.path).split('.')[0]
                        frame_id = int(frame_idx)
                    else:
                        continue

                    if result.boxes.id is None:
                        continue

                    boxes = result.boxes.xywh.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()

                    for box, track_id, conf in zip(boxes, track_ids, confs):
                        x_c, y_c, w, h = box
                        x1 = x_c - (w / 2)
                        y1 = y_c - (h / 2)

                        line = f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
                        f.write(line)

        print("\n--- Tracking Completato ---")