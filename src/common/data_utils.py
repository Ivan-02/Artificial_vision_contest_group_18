import configparser
import os

class GameInfoParser:
    @staticmethod
    def get_id_map(ini_path):
        """Restituisce mappa {id: label_description}."""
        id_map = {}
        if not os.path.exists(ini_path): return id_map
        try:
            config = configparser.ConfigParser()
            config.read(ini_path)
            if 'Sequence' in config:
                for key, val in config['Sequence'].items():
                    if key.startswith('trackletid_'):
                        try:
                            obj_id = int(key.split('_')[1])
                            id_map[obj_id] = val.split(';')[0].lower().strip()
                        except:
                            continue
        except Exception:
            pass
        return id_map

    @staticmethod
    def get_ball_id(ini_path):
        """Restituisce l'ID della palla se presente."""
        mapping = GameInfoParser.get_id_map(ini_path)
        for oid, label in mapping.items():
            if 'ball' in label:
                return oid
        return None


class GeometryUtils:
    @staticmethod
    def is_in_roi(box_xywh, roi_norm, img_w, img_h):
        """Verifica se il centro inferiore del box è nella ROI normalizzata."""
        x_c, y_c, w_box, h_box = box_xywh
        base_x = x_c
        base_y = y_c + (h_box / 2.0)  # Piedi del giocatore

        rx = roi_norm['x'] * img_w
        ry = roi_norm['y'] * img_h
        rw = roi_norm['width'] * img_w
        rh = roi_norm['height'] * img_h

        return (rx <= base_x <= rx + rw) and (ry <= base_y <= ry + rh)

    @staticmethod
    def calculate_iou(box1, box2):
        """Calcola Intersection over Union standard."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0