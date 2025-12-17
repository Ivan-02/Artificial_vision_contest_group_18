import cv2
import numpy as np

class Visualizer:
    def __init__(self, display_width=None):
        self.display_width = display_width
        self.colors = {}

    def get_color(self, id_int):
        if id_int not in self.colors:
            np.random.seed(int(id_int))
            self.colors[id_int] = tuple(np.random.randint(0, 255, size=3).tolist())
        return self.colors[id_int]

    def draw_box(self, img, box_xyxy, track_id=None, cls_id=None, conf=None, color=None):
        x1, y1, x2, y2 = map(int, box_xyxy)
        c = color if color else self.get_color(track_id if track_id is not None else 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)

        if track_id is not None:
            label = f"ID:{track_id}"
            if conf is not None: label += f" {conf:.2f}"

            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w_text, y1), c, -1)
            text_color = (255, 255, 255) if sum(c) < 400 else (0, 0, 0)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    def draw_roi(self, img, roi_def, count, color=(0, 255, 255), label_key="ROI"):
        h, w = img.shape[:2]
        x = int(roi_def['x'] * w)
        y = int(roi_def['y'] * h)
        rw = int(roi_def['width'] * w)
        rh = int(roi_def['height'] * h)

        cv2.rectangle(img, (x, y), (x + rw, y + rh), color, 2)

        text = f"{label_key}: {count}"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def show_frame(self, window_name, img):
        if self.display_width:
            h, w = img.shape[:2]
            aspect = h / w
            d_h = int(self.display_width * aspect)
            img = cv2.resize(img, (self.display_width, d_h))

        cv2.imshow(window_name, img)
        return cv2.waitKey(1) & 0xFF != ord('q')

    def close_windows(self):
        cv2.destroyAllWindows()