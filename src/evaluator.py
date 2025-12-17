import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import re
import json  # <--- IMPORT NECESSARIO


class Evaluator:
    def __init__(self, config):
        self.cfg = config
        self.alpha = 0.5
        self.base_output_dir = self.cfg['paths']['output_submission']
        self.subdirs = self.cfg['paths']['output_subdirs']
        self.gt_base_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['paths']['split'])
        self.team_name = self.cfg['names']['team']  # Es. 18

    def _build_gt_map(self, mode):
        gt_map = {}
        if not os.path.exists(self.gt_base_dir):
            return gt_map

        for video_folder in os.listdir(self.gt_base_dir):
            gt_folder_path = os.path.join(self.gt_base_dir, video_folder, 'gt')
            if not os.path.isdir(gt_folder_path):
                continue

            if mode == 'behavior':
                for f in os.listdir(gt_folder_path):
                    if f.startswith("behavior_") and f.endswith("_gt.txt"):
                        parts = f.split('_')
                        if len(parts) >= 3:
                            vid_id = parts[1]
                            gt_map[vid_id] = os.path.join(gt_folder_path, f)

            elif mode == 'tracking':
                if 'gt.txt' in os.listdir(gt_folder_path):
                    match = re.search(r'(\d+)$', video_folder)
                    if match:
                        vid_id = match.group(1)
                    else:
                        vid_id = video_folder

                    gt_map[vid_id] = os.path.join(gt_folder_path, 'gt.txt')

        return gt_map

    def run_behavior(self):
        print("\n" + "=" * 60)
        print("AVVIO VALUTAZIONE: BEHAVIOR ANALYSIS (nMAE)")
        print("=" * 60)

        gt_map = self._build_gt_map('behavior')

        for subdir in self.subdirs:
            print(f"\n--- Elaborazione Behavior directory: {subdir} ---")

            pred_dir = os.path.join(self.base_output_dir, subdir)
            metrics_dir = pred_dir
            os.makedirs(metrics_dir, exist_ok=True)

            # --- SETUP LETTURA E AGGIORNAMENTO JSON (BEHAVIOR) ---
            json_report_path = os.path.join(metrics_dir, "execution_report.json")
            json_data = {}
            if os.path.exists(json_report_path):
                try:
                    with open(json_report_path, 'r') as f:
                        json_data = json.load(f)
                except Exception as e:
                    print(f"Attenzione: Impossibile leggere il JSON esistente: {e}")

            if "video_scores" not in json_data:
                json_data["video_scores"] = {}
            # -----------------------------------------------------

            if not os.path.exists(pred_dir):
                print(f"Errore: Cartella predizioni non trovata: {pred_dir}")
                continue

            # Cerchiamo file del tipo behavior_XX_TEAM.txt
            pred_files = [f for f in os.listdir(pred_dir)
                          if f.startswith("behavior_") and f.endswith(f"_{self.team_name}.txt")]

            if not pred_files:
                print(f"Nessun file behavior trovato in {subdir} per il team {self.team_name}.")
                continue

            total_abs_error = 0.0
            total_samples = 0
            video_stats = []

            for pred_file in pred_files:
                # Estrazione ID video dal nome file: behavior_XX_18.txt
                try:
                    video_id = pred_file.split('_')[1]
                except IndexError:
                    print(f"Skipping file con nome non valido: {pred_file}")
                    continue

                if video_id not in gt_map:
                    print(
                        f"GT non trovata per video ID {video_id} (File cercato: behavior_{video_id}_gt.txt). Skipping.")
                    continue

                gt_path = gt_map[video_id]
                pred_path = os.path.join(pred_dir, pred_file)

                # Caricamento dati
                try:
                    df_gt = pd.read_csv(gt_path, header=None, names=['frame_id', 'region_id', 'n_players'])
                    df_pred = pd.read_csv(pred_path, header=None, names=['frame_id', 'region_id', 'n_players'])
                except Exception as e:
                    print(f"Errore lettura file per video {video_id}: {e}")
                    continue

                # Merge per allineare frame e region.
                merged = pd.merge(df_gt, df_pred, on=['frame_id', 'region_id'], how='left', suffixes=('_gt', '_pred'))
                merged['n_players_pred'] = merged['n_players_pred'].fillna(0)
                merged['abs_err'] = abs(merged['n_players_gt'] - merged['n_players_pred'])

                vid_samples = len(merged)
                vid_err = merged['abs_err'].sum()
                vid_mae = vid_err / vid_samples if vid_samples > 0 else 0

                # Calcolo nMAE specifico per il video
                vid_nmae = (10 - min(10, vid_mae)) / 10

                video_stats.append({
                    'VideoID': video_id,
                    'Samples': vid_samples,
                    'AbsError': vid_err,
                    'MAE': round(vid_mae, 4)
                })

                # --- AGGIORNAMENTO JSON PER QUESTO VIDEO ---
                # Logica per trovare la chiave corretta (es. 'test-01' da '01')
                json_key_match = None
                if "video_execution_times" in json_data:
                    for key in json_data["video_execution_times"].keys():
                        if video_id in key:
                            json_key_match = key
                            break

                if not json_key_match:
                    json_key_match = video_id

                # Assicuriamoci che la chiave esista in video_scores
                if json_key_match not in json_data["video_scores"]:
                    json_data["video_scores"][json_key_match] = {}

                # Aggiungiamo le metriche
                json_data["video_scores"][json_key_match]["MAE"] = round(vid_mae, 4)
                json_data["video_scores"][json_key_match]["nMAE"] = round(vid_nmae, 4)
                # -------------------------------------------

                total_abs_error += vid_err
                total_samples += vid_samples

            # Calcolo MAE Globale e nMAE finale
            if total_samples > 0:
                global_mae = total_abs_error / total_samples
                nmae = (10 - min(10, global_mae)) / 10
            else:
                global_mae = 0.0
                nmae = 0.0

            # Salvataggio Report CSV
            df_res = pd.DataFrame(video_stats)
            avg_row = {
                'VideoID': 'GLOBAL',
                'Samples': total_samples,
                'AbsError': total_abs_error,
                'MAE': round(global_mae, 4)
            }
            df_res = pd.concat([df_res, pd.DataFrame([avg_row])], ignore_index=True)

            csv_path = os.path.join(metrics_dir, 'behavior_metrics.csv')
            df_res.to_csv(csv_path, index=False)

            # --- SALVATAGGIO JSON FINALE ---
            try:
                with open(json_report_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
                print(f"Report JSON aggiornato con nMAE salvato: {json_report_path}")
            except Exception as e:
                print(f"Errore nel salvataggio del JSON: {e}")
            # -------------------------------

            print(f"[{subdir}] GLOBAL MAE: {global_mae:.4f} | nMAE SCORE: {nmae:.4f}")
            print(f"Report CSV salvato: {csv_path}")

    def run_hota(self):
        print("\n" + "=" * 60)
        print("AVVIO VALUTAZIONE: TRACKING (HOTA)")
        print("=" * 60)

        gt_map = self._build_gt_map(mode='tracking')

        for subdir in self.subdirs:
            print(f"\n--- Elaborazione HOTA directory: {subdir} ---")

            pred_dir = os.path.join(self.base_output_dir, subdir)
            metrics_dir = pred_dir
            os.makedirs(metrics_dir, exist_ok=True)

            # --- SETUP LETTURA E AGGIORNAMENTO JSON ---
            json_report_path = os.path.join(metrics_dir, "execution_report.json")
            json_data = {}
            if os.path.exists(json_report_path):
                try:
                    with open(json_report_path, 'r') as f:
                        json_data = json.load(f)
                except Exception as e:
                    print(f"Attenzione: Impossibile leggere il JSON esistente: {e}")

            if "video_scores" not in json_data:
                json_data["video_scores"] = {}
            # -------------------------------------------

            all_videos_metrics = []
            global_stats = {
                'TP': 0, 'FP': 0, 'FN': 0,
                'AssA_sum': 0.0, 'TP_count': 0
            }

            if not os.path.exists(pred_dir):
                print(f"Errore: Cartella predizioni non trovata: {pred_dir}")
                continue

            pred_files = [f for f in os.listdir(pred_dir)
                          if f.endswith('.txt') and not f.startswith('behavior_')]

            if not pred_files:
                print(f"Nessun file tracking trovato in {subdir}.")
                continue

            print(f"Trovati {len(pred_files)} video da valutare in {subdir}.")

            for pred_file in pred_files:
                video_name_clean = pred_file.replace('.txt', '')

                # Estrazione ID dal nome file predizione
                # tracking_01_team.txt -> 01
                try:
                    vid_id_str = video_name_clean.split("_")[1]
                except IndexError:
                    vid_id_str = video_name_clean

                pred_path = os.path.join(pred_dir, pred_file)

                if vid_id_str in gt_map:
                    gt_path = gt_map[vid_id_str]
                else:
                    print(f"GT non trovata per Video ID: {vid_id_str}. Skipping.")
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
                    f"[{subdir}] Video: {vid_id_str:<10} | HOTA: {hota_vid:.4f} | DetA: {deta_vid:.4f} | AssA: {assa_vid:.4f}")

                # --- AGGIORNAMENTO JSON ---
                # Cerchiamo la chiave corretta nel JSON (che usa nomi tipo 'test-01')
                # confrontandola con vid_id_str ('01')
                json_key_match = None
                if "video_execution_times" in json_data:
                    for key in json_data["video_execution_times"].keys():
                        # Se l'ID (es. '01') è contenuto nella chiave (es. 'test-01')
                        if vid_id_str in key:
                            json_key_match = key
                            break

                # Se non trovato nel dizionario tempi, usiamo l'ID come chiave
                if not json_key_match:
                    json_key_match = vid_id_str

                if json_key_match not in json_data["video_scores"]:
                    json_data["video_scores"][json_key_match] = {}

                json_data["video_scores"][json_key_match].update({
                    "HOTA": round(hota_vid, 4),
                    "DetA": round(deta_vid, 4),
                    "AssA": round(assa_vid, 4)
                })
                # --------------------------

                video_row = {
                    'Video': vid_id_str,
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
                    'Video': 'GLOBAL_SCORE',
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
                csv_path = os.path.join(metrics_dir, 'hota_metrics.csv')
                df_final.to_csv(csv_path, index=False)

                # --- SALVATAGGIO JSON FINALE ---
                try:
                    with open(json_report_path, 'w') as f:
                        json.dump(json_data, f, indent=4)
                    print(f"Report JSON aggiornato salvato: {json_report_path}")
                except Exception as e:
                    print(f"Errore nel salvataggio del JSON: {e}")
                # -------------------------------

                print(f"Report HOTA CSV salvato: {os.path.abspath(csv_path)}")

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

    @staticmethod
    def _get_ball_id(gameinfo_path):
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