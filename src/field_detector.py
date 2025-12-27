import cv2
import numpy as np


class FieldFilter:
    """
    Gestisce la segmentazione del campo da gioco e il filtraggio delle rilevazioni,
    utilizzando maschere HSV, operazioni morfologiche e analisi geometrica dei contorni.
    """

    def __init__(self, settings=None):
        """
        Inizializza i parametri di filtraggio, definendo i range di colore HSV,
        i kernel per la morfologia e le soglie per la validazione delle coordinate.
        """
        if settings is None:
            settings = {}

        self.debug = settings.get('debug', True)

        hsv_lower = settings.get('hsv_lower', [35, 27, 40])
        hsv_upper = settings.get('hsv_upper', [95, 200, 200])
        self.lower_green = np.array(hsv_lower, dtype=np.uint8)
        self.upper_green = np.array(hsv_upper, dtype=np.uint8)

        k_erode = settings.get('kernel_erode_size', 15)
        k_dilate = settings.get('kernel_dilate_size', 13)
        k_close = settings.get('kernel_close_size', 30)
        self.morph_iterations = settings.get('morph_iterations', 2)

        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))

        self.alpha = settings.get('alpha_smooth', 0.7)
        self.binary_threshold = settings.get('binary_threshold', 100)
        self.min_area_ratio = settings.get('min_area_ratio', 0.01)
        self.poly_epsilon = settings.get('poly_epsilon', 0.01)
        self.show_mosaic = settings.get('debug_mosaic', False)

        self.pixel_check_radius = settings.get('pixel_check_radius', 5)
        self.pixel_check_ratio = settings.get('pixel_check_ratio', 0.1)
        self.bottom_zone_ratio = settings.get('bottom_zone_ratio', 0.80)
        self.strong_conf_thresh = settings.get('strong_conf_thresh', 0.75)
        self.safe_zone_dist = settings.get('safe_zone_dist', 30)

        self.clipping_margin = settings.get('clipping_margin', 5)
        self.pixel_check_offset = settings.get('pixel_check_offset', 10)

        self.thresh_clipping = settings.get('thresh_clipping', -5.0)
        self.thresh_bottom = settings.get('thresh_bottom', 5.0)
        self.thresh_std = settings.get('thresh_standard', -2.0)

        self.prev_mask = None
        self.current_binary_mask = None

        self.step1_hsv = None
        self.step2_morph = None
        self.step3_filled = None

    def _get_field_contour(self, frame):
        """
        Elabora il frame per estrarre il contorno del campo tramite soglie di colore,
        accumulazione pesata per la stabilità temporale e operazioni di chiusura/erosione.
        """
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        curr_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        if self.debug: self.step1_hsv = curr_mask.copy()

        if self.prev_mask is None:
            self.prev_mask = curr_mask.astype(np.float32)
        else:
            cv2.accumulateWeighted(curr_mask, self.prev_mask, 1 - self.alpha)

        _, processed_mask = cv2.threshold(self.prev_mask, self.binary_threshold, 255, cv2.THRESH_BINARY)
        processed_mask = processed_mask.astype(np.uint8)

        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, self.kernel_close)
        processed_mask = cv2.erode(processed_mask, self.kernel_erode, iterations=self.morph_iterations)
        processed_mask = cv2.dilate(processed_mask, self.kernel_dilate, iterations=self.morph_iterations)

        if self.debug: self.step2_morph = processed_mask.copy()

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.current_binary_mask = processed_mask
            if self.debug: self.step3_filled = processed_mask.copy()
            return None

        min_area = (frame.shape[0] * frame.shape[1] * self.min_area_ratio)
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not significant_contours:
            self.current_binary_mask = processed_mask
            if self.debug: self.step3_filled = processed_mask.copy()
            return None

        all_points = np.vstack(significant_contours)
        hull = cv2.convexHull(all_points)
        epsilon = self.poly_epsilon * cv2.arcLength(hull, True)
        approx_contour = cv2.approxPolyDP(hull, epsilon, True)

        filled_mask = np.zeros_like(processed_mask)
        cv2.drawContours(filled_mask, [hull], -1, 255, thickness=cv2.FILLED)

        if self.debug: self.step3_filled = filled_mask.copy()
        self.current_binary_mask = filled_mask

        return approx_contour

    def _check_pixel_under_feet(self, x, y, frame_w, frame_h):
        """
        Esegue un controllo puntuale sulla maschera binaria per verificare se l'area
        sottostante a una coordinata specifica appartiene effettivamente al campo.
        """
        if self.current_binary_mask is None: return True
        r = self.pixel_check_radius
        x1, x2 = max(0, x - r), min(frame_w, x + r)
        y1, y2 = max(0, y - r), min(frame_h, y + r)
        roi = self.current_binary_mask[y1:y2, x1:x2]
        if roi.size == 0: return False
        return (cv2.countNonZero(roi) / roi.size) > self.pixel_check_ratio

    def filter_detections(self, frame, detections):
        """
        Analizza ogni rilevazione confrontando la posizione dei piedi con il contorno
        del campo e la presenza di colore verde, filtrando gli oggetti esterni al gioco.
        """
        field_contour = self._get_field_contour(frame)

        if field_contour is None:
            return detections, None

        filtered_detections = []
        img_h, img_w = frame.shape[:2]

        bottom_limit_y = img_h * self.bottom_zone_ratio

        for det in detections:
            xc, yc, w, h = det['xywh']
            feet_x = int(xc)
            feet_y = int(yc + (h / 2.0))

            dist = cv2.pointPolygonTest(field_contour, (float(feet_x), float(feet_y)), True)

            is_clipping_bottom = feet_y >= (img_h - self.clipping_margin)

            if is_clipping_bottom:
                threshold = self.thresh_clipping
            elif feet_y > bottom_limit_y:
                threshold = self.thresh_bottom
            else:
                threshold = self.thresh_std

            is_geometrically_inside = dist >= threshold

            if is_clipping_bottom:
                safe_y = max(0, feet_y - self.pixel_check_offset)
                is_ground_green = self._check_pixel_under_feet(feet_x, safe_y, img_w, img_h)
            else:
                is_ground_green = self._check_pixel_under_feet(feet_x, feet_y, img_w, img_h)

            strong_detection = det['conf'] > self.strong_conf_thresh
            condition_safe_zone = (dist > self.safe_zone_dist)

            if is_geometrically_inside and (is_ground_green or condition_safe_zone or (is_clipping_bottom and strong_detection)):
                filtered_detections.append(det)
                final_decision = True
            else:
                final_decision = False

            if self.debug:
                color = (0, 255, 0) if final_decision else (0, 0, 255)
                cv2.circle(frame, (feet_x, feet_y), 5, color, -1)
                if feet_y > bottom_limit_y:
                    info = f"Cut:{int(is_clipping_bottom)}"
                    cv2.putText(frame, info, (feet_x - 20, feet_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return filtered_detections, field_contour

    def draw_debug(self, frame, contour):
        """
        Genera una visualizzazione grafica per il debug, mostrando il perimetro del campo
        o un mosaico dei passaggi intermedi della segmentazione (HSV, Morph, Filled).
        """
        if not self.debug:
            return frame

        vis_frame = frame.copy()

        if contour is not None:
            cv2.drawContours(vis_frame, [contour], -1, (0, 255, 0), 2)
            if not self.show_mosaic:
                cv2.putText(vis_frame, "FIELD FILTER ACTIVE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not self.show_mosaic:
            return vis_frame

        def prepare_mask(mask, label):
            if mask is None:
                blank = np.zeros_like(frame)
                cv2.putText(blank, "N/A", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return blank
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_bgr, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return mask_bgr

        img_hsv = prepare_mask(self.step1_hsv, "STEP 1: HSV (Raw)")
        img_morph = prepare_mask(self.step2_morph, "STEP 2: Morfologia")
        img_filled = prepare_mask(self.step3_filled, "STEP 3: Hole Filling (Fix)")

        h, w = frame.shape[:2]
        new_dim = (int(w * 0.5), int(h * 0.5))

        top_left = cv2.resize(vis_frame, new_dim)
        top_right = cv2.resize(img_hsv, new_dim)
        bot_left = cv2.resize(img_morph, new_dim)
        bot_right = cv2.resize(img_filled, new_dim)

        return np.vstack((np.hstack((top_left, top_right)), np.hstack((bot_left, bot_right))))