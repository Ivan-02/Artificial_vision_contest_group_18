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

            feet_x = int(xc)
            feet_y = int(yc + (h / 2.0))

            dist = cv2.pointPolygonTest(field_contour, (float(feet_x), float(feet_y)), True)

            if feet_y > (img_h * 0.80):
                threshold = 5.0
            else:
                threshold = -15.0

            is_geometrically_inside = dist >= threshold

            is_ground_green = self._check_pixel_under_feet(feet_x, feet_y, img_w, img_h)

            if is_geometrically_inside and (is_ground_green or dist > 50):
                filtered_detections.append(det)
                final_decision = True
            else:
                final_decision = False

            if self.debug:
                color = (0, 255, 0) if final_decision else (0, 0, 255)
                cv2.circle(frame, (feet_x, feet_y), 5, color, -1)

        return filtered_detections, field_contour

    def draw_debug(self, frame, contour):
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        return frame