import cv2
import numpy as np


class FieldFilter:
    def __init__(self, debug=False):
        self.debug = debug

        self.lower_green = np.array([35, 40, 45])
        self.upper_green = np.array([90, 255, 255])

        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

        self.prev_mask = None
        self.alpha = 0.7
        self.current_binary_mask = None

    def _get_field_contour(self, frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        curr_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        if self.prev_mask is None:
            self.prev_mask = curr_mask.astype(np.float32)
        else:
            cv2.accumulateWeighted(curr_mask, self.prev_mask, 1 - self.alpha)

        _, processed_mask = cv2.threshold(self.prev_mask, 100, 255, cv2.THRESH_BINARY)
        processed_mask = processed_mask.astype(np.uint8)

        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, self.kernel_close)

        processed_mask = cv2.erode(processed_mask, self.kernel_erode, iterations=2)
        processed_mask = cv2.dilate(processed_mask, self.kernel_dilate, iterations=2)

        self.current_binary_mask = processed_mask

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours: return None

        min_area = (frame.shape[0] * frame.shape[1] * 0.01)
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not significant_contours: return None

        all_points = np.vstack(significant_contours)
        hull = cv2.convexHull(all_points)
        epsilon = 0.01 * cv2.arcLength(hull, True)
        approx_contour = cv2.approxPolyDP(hull, epsilon, True)

        return approx_contour

    def _check_pixel_under_feet(self, x, y, frame_w, frame_h):
        if self.current_binary_mask is None: return True
        r = 5
        x1, x2 = max(0, x - r), min(frame_w, x + r)
        y1, y2 = max(0, y - r), min(frame_h, y + r)
        roi = self.current_binary_mask[y1:y2, x1:x2]
        if roi.size == 0: return False
        return (cv2.countNonZero(roi) / roi.size) > 0.3

    def filter_detections(self, frame, detections):
        field_contour = self._get_field_contour(frame)

        if field_contour is None:
            return detections, None

        filtered_detections = []
        img_h, img_w = frame.shape[:2]

        for det in detections:
            xc, yc, w, h = det['xywh']

            # Calcolo coordinate piedi
            feet_x = int(xc)
            feet_y = int(yc + (h / 2.0))

            # Distanza dal contorno (+ dentro, - fuori)
            dist = cv2.pointPolygonTest(field_contour, (float(feet_x), float(feet_y)), True)

            # --- LOGICA CORRETTA ---

            # 1. Controlliamo se il box tocca il bordo inferiore dell'immagine (con margine di 5px)
            # Questo identifica i giocatori "tagliati" dalla telecamera.
            is_clipping_bottom = feet_y >= (img_h - 5)

            if is_clipping_bottom:
                # CASO: Giocatore tagliato a metà.
                # Non possiamo essere severi (+5) perché il bordo del campo coincide col bordo immagine.
                # Accettiamo anche 0 o leggermente fuori (-5) per compensare l'erosione.
                threshold = -5.0

            elif feet_y > (img_h * 0.80):
                # CASO: Parte bassa ma NON tagliato (Vigili, Allenatori).
                # Qui rimaniamo SEVERI per evitare i falsi positivi sulla pista.
                threshold = 5.0

            else:
                # CASO: Parte alta/centrale del campo.
                threshold = -15.0

            # Controllo Geometrico
            is_geometrically_inside = dist >= threshold

            # Controllo Colore (Pixel Check)
            # Se è tagliato in basso, il controllo colore sui piedi potrebbe fallire (siamo al bordo).
            # Ci fidiamo della geometria (Convex Hull) che in quel punto dovrebbe coprire tutto.
            if is_clipping_bottom:
                # Per chi è tagliato, controlliamo il colore un po' più in su (es. ginocchia)
                # per evitare artefatti del bordo immagine, oppure ci fidiamo solo della geometria.
                # Qui controllo 10px sopra il bordo.
                safe_y = max(0, feet_y - 10)
                is_ground_green = self._check_pixel_under_feet(feet_x, safe_y, img_w, img_h)
            else:
                is_ground_green = self._check_pixel_under_feet(feet_x, feet_y, img_w, img_h)

            # Decisione Finale
            # Nota: Se è tagliato (is_clipping_bottom), diamo priorità alla geometria
            # perché il colore al bordo estremo è spesso instabile.
            if is_geometrically_inside and (is_ground_green or dist > 50 or is_clipping_bottom):
                filtered_detections.append(det)
                final_decision = True
            else:
                final_decision = False

            if self.debug:
                # Debug visivo migliorato
                color = (0, 255, 0) if final_decision else (0, 0, 255)
                cv2.circle(frame, (feet_x, feet_y), 5, color, -1)

                # Scrivi info di debug se siamo in basso
                if feet_y > img_h * 0.8:
                    info = f"Cut:{int(is_clipping_bottom)} D:{int(dist)}"
                    cv2.putText(frame, info, (feet_x - 20, feet_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return filtered_detections, field_contour

    def draw_debug(self, frame, contour):
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        return frame