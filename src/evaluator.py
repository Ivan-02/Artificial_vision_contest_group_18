import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class HotaEvaluator:
    def __init__(self, config):
        self.cfg = config
        self.alpha = 0.5
        self.base_output_dir = self.cfg['paths']['output_submission']
        self.subdirs = self.cfg['paths']['output_subdirs']

    def run(self):
        gt_base_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['paths']['split'])

        for subdir in self.subdirs:
            print(f"\n--- Elaborazione directory: {subdir} ---")

            pred_dir = os.path.join(self.base_output_dir, subdir)
            metrics_dir = pred_dir
            os.makedirs(metrics_dir, exist_ok=True)

            all_videos_metrics = []

            global_stats = {
                'TP': 0,
                'FP': 0,
                'FN': 0,
                'AssA_sum': 0.0,
                'TP_count': 0
            }

            if not os.path.exists(pred_dir):
                print(f"Errore: Cartella predizioni non trovata: {pred_dir}")
                continue

            pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]

            if not pred_files:
                print(f"Nessun file di predizione trovato in {subdir}.")
                continue

            print(f"Trovati {len(pred_files)} video da valutare in {subdir}.")

            for pred_file in pred_files:
                video_name = pred_file.replace('.txt', '')
                video_name = f"SNMOT-{video_name.split("_")[1]}"
                pred_path = os.path.join(pred_dir, pred_file)

                gt_path = os.path.join(gt_base_dir, video_name, 'gt', 'gt.txt')

                if not os.path.exists(gt_path):
                    print(f"GT non trovata per {video_name}. Skipping.")
                    continue

                v_stats = self._evaluate_video(gt_path, pred_path)

                tp = v_stats['TP']
                fn = v_stats['FN']
                fp = v_stats['FP']

                avg_tpa = v_stats['TPA_sum'] / tp if tp > 0 else 0
                avg_fna = v_stats['FNA_sum'] / tp if tp > 0 else 0
                avg_fpa = v_stats['FPA_sum'] / tp if tp > 0 else 0

                deta_vid = self._calc_deta(tp, fn, fp)
                assa_vid = self._calc_assa(v_stats['AssA_sum'], tp)
                hota_vid = np.sqrt(deta_vid * assa_vid)

                print(
                    f"[{subdir}] Video: {video_name:<20} | HOTA: {hota_vid:.4f} | DetA: {deta_vid:.4f} | AssA: {assa_vid:.4f}")

                video_row = {
                    'Video': video_name,
                    'HOTA': round(hota_vid, 4),
                    'DetA': round(deta_vid, 4),
                    'AssA': round(assa_vid, 4),
                    'TP': tp,
                    'FN': fn,
                    'FP': fp,
                    'Avg_TPA': round(avg_tpa, 2),
                    'Avg_FNA': round(avg_fna, 2),
                    'Avg_FPA': round(avg_fpa, 2),
                    'Sum_AssA_Score': round(v_stats['AssA_sum'], 4)
                }
                all_videos_metrics.append(video_row)

                global_stats['TP'] += tp
                global_stats['FP'] += fp
                global_stats['FN'] += fn
                global_stats['AssA_sum'] += v_stats['AssA_sum']
                global_stats['TP_count'] += tp

            final_deta = self._calc_deta(global_stats['TP'], global_stats['FN'], global_stats['FP'])
            final_assa = self._calc_assa(global_stats['AssA_sum'], global_stats['TP_count'])
            final_hota = np.sqrt(final_deta * final_assa)

            if all_videos_metrics:
                df = pd.DataFrame(all_videos_metrics)

                avg_row = {
                    'Video': 'GLOBAL_SCORE (Weighted)',
                    'HOTA': round(final_hota, 4),
                    'DetA': round(final_deta, 4),
                    'AssA': round(final_assa, 4),
                    'TP': global_stats['TP'],
                    'FN': global_stats['FN'],
                    'FP': global_stats['FP'],
                    'Avg_TPA': round(df['Avg_TPA'].mean(), 2),
                    'Avg_FNA': round(df['Avg_FNA'].mean(), 2),
                    'Avg_FPA': round(df['Avg_FPA'].mean(), 2),
                    'Sum_AssA_Score': round(global_stats['AssA_sum'], 4)
                }

                df_final = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

                csv_path = os.path.join(metrics_dir, 'detailed_metrics.csv')
                df_final.to_csv(csv_path, index=False)
                print(f"Report salvato per {subdir}: {os.path.abspath(csv_path)}")

            print("-" * 60)
            print(f"RISULTATO {subdir}: HOTA {final_hota:.4f}")
            print("-" * 60)

    def _evaluate_video(self, gt_path, pred_path):
        video_dir = os.path.dirname(os.path.dirname(gt_path))
        gameinfo_path = os.path.join(video_dir, 'gameinfo.ini')
        ball_id = self._get_ball_id(gameinfo_path)

        gt_data = self._load_data(gt_path, ignore_id=ball_id)
        pred_data = self._load_data(pred_path)

        all_frames = sorted(list(set(gt_data.keys()) | set(pred_data.keys())))

        tp_count = 0
        fp_count = 0
        fn_count = 0
        matches_list = []

        for frame_id in all_frames:
            gts = gt_data.get(frame_id, [])
            preds = pred_data.get(frame_id, [])

            iou_matrix = np.zeros((len(gts), len(preds)))
            for i, gt in enumerate(gts):
                for j, pred in enumerate(preds):
                    iou_matrix[i, j] = self._calculate_iou(gt['box'], pred['box'])

            if iou_matrix.size > 0:
                cost_matrix = 1 - iou_matrix
                cost_matrix[iou_matrix < self.alpha] = 1000.0

                gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

                valid_matches_count = 0
                for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                    if iou_matrix[gt_idx, pred_idx] >= self.alpha:
                        valid_matches_count += 1
                        g_id = gts[gt_idx]['obj_id']
                        p_id = preds[pred_idx]['obj_id']
                        matches_list.append((g_id, p_id))

                tp_count += valid_matches_count
                fn_count += len(gts) - valid_matches_count
                fp_count += len(preds) - valid_matches_count

            else:
                fn_count += len(gts)
                fp_count += len(preds)

        match_counts = defaultdict(int)
        for m in matches_list:
            match_counts[m] += 1

        gt_id_counts = defaultdict(int)
        pred_id_counts = defaultdict(int)

        for gt_id, pred_id in matches_list:
            gt_id_counts[gt_id] += 1
            pred_id_counts[pred_id] += 1

        assa_sum = 0.0
        tpa_sum_vid = 0
        fna_sum_vid = 0
        fpa_sum_vid = 0

        for gt_id, pred_id in matches_list:
            tpa = match_counts[(gt_id, pred_id)]
            fna = gt_id_counts[gt_id] - tpa
            fpa = pred_id_counts[pred_id] - tpa

            a_p = tpa / (tpa + fna + fpa)
            assa_sum += a_p

            tpa_sum_vid += tpa
            fna_sum_vid += fna
            fpa_sum_vid += fpa

        return {
            'TP': tp_count,
            'FP': fp_count,
            'FN': fn_count,
            'AssA_sum': assa_sum,
            'TPA_sum': tpa_sum_vid,
            'FNA_sum': fna_sum_vid,
            'FPA_sum': fpa_sum_vid
        }

    def _get_ball_id(self, gameinfo_path):
        if not os.path.exists(gameinfo_path):
            return None

        try:
            with open(gameinfo_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('trackletID_'):
                        parts = line.split('=')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].lower()

                            if 'ball' in value:
                                try:
                                    tid_str = key.split('_')[1]
                                    return int(tid_str)
                                except (IndexError, ValueError):
                                    continue
        except Exception:
            pass
        return None

    @staticmethod
    def _calc_deta(tp, fn, fp):
        den = tp + fn + fp
        if den == 0: return 0.0
        return tp / den

    @staticmethod
    def _calc_assa(assa_sum, tp_count):
        if tp_count == 0: return 0.0
        return assa_sum / tp_count

    @staticmethod
    def _load_data(file_path, ignore_id=None):
        data = defaultdict(list)
        try:
            df = pd.read_csv(file_path, header=None)
            vals = df.iloc[:, :6].values

            for row in vals:
                frame_id = int(row[0])
                obj_id = int(row[1])
                x, y, w, h = row[2], row[3], row[4], row[5]

                if obj_id < 0: continue
                if ignore_id is not None and obj_id == ignore_id: continue

                data[frame_id].append({
                    'obj_id': obj_id,
                    'box': [x, y, w, h]
                })
        except Exception:
            pass
        return data

    @staticmethod
    def _calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]

        union = area1 + area2 - intersection
        if union <= 0: return 0.0
        return intersection / union