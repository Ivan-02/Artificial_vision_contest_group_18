import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class HotaEvaluator:
    def __init__(self, config):
        self.cfg = config
        self.alpha = 0.5  # Soglia IoU definita nel PDF

    def run(self):
        """
        Esegue la valutazione confrontando i file di Ground Truth con le predizioni.
        Assume che le predizioni siano già state generate in output_submission.
        """
        print("\n--- Avvio Valutazione HOTA0.5 ---")

        # Percorsi
        gt_base_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['val']['split'])
        pred_dir = self.cfg['paths']['output_submission']

        # Accumulatori globali per tutti i video
        global_stats = {
            'TP': 0,
            'FP': 0,
            'FN': 0,
            'AssA_sum': 0.0,  # Somma dei punteggi di associazione per i TP
            'TP_count': 0  # Numero totale di TP (uguale a TP sopra, ma usato per AssA)
        }

        # Trova i video disponibili nelle predizioni
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]

        if not pred_files:
            print("Nessun file di predizione trovato. Esegui prima il Tracker.")
            return

        metrics_per_video = []

        for pred_file in pred_files:
            video_name = pred_file.replace('.txt', '')
            pred_path = os.path.join(pred_dir, pred_file)

            # Costruisci percorso GT (assume struttura standard SoccerNet/MOT)
            gt_path = os.path.join(gt_base_dir, video_name, 'gt', 'gt.txt')

            if not os.path.exists(gt_path):
                print(f"Ground Truth non trovata per {video_name}. Skipping.")
                continue

            # Calcola metriche per il singolo video
            video_stats = self._evaluate_video(gt_path, pred_path)

            # Calcolo HOTA parziale per display
            deta_vid = self._calc_deta(video_stats['TP'], video_stats['FN'], video_stats['FP'])
            assa_vid = self._calc_assa(video_stats['AssA_sum'], video_stats['TP'])
            hota_vid = np.sqrt(deta_vid * assa_vid)

            print(f"Video: {video_name} | DetA: {deta_vid:.4f} | AssA: {assa_vid:.4f} | HOTA: {hota_vid:.4f}")

            # Aggiorna globali
            global_stats['TP'] += video_stats['TP']
            global_stats['FP'] += video_stats['FP']
            global_stats['FN'] += video_stats['FN']
            global_stats['AssA_sum'] += video_stats['AssA_sum']

        # --- CALCOLO FINALE HOTA0.5  ---
        final_deta = self._calc_deta(global_stats['TP'], global_stats['FN'], global_stats['FP'])
        final_assa = self._calc_assa(global_stats['AssA_sum'], global_stats['TP'])
        final_hota = np.sqrt(final_deta * final_assa)

        print("=" * 60)
        print(f"RISULTATO FINALE (Media sui video):")
        print(f"DetA (Detection Accuracy): {final_deta:.4f}")
        print(f"AssA (Association Accuracy): {final_assa:.4f}")
        print(f"HOTA0.5 Score:             {final_hota:.4f}")
        print("=" * 60)

        return final_hota

    def _evaluate_video(self, gt_path, pred_path):
        """Elabora un singolo video frame per frame."""

        # Carica dati
        gt_data = self._load_data(gt_path)
        pred_data = self._load_data(pred_path)

        # Identifica tutti i frame unici
        all_frames = sorted(list(set(gt_data.keys()) | set(pred_data.keys())))

        # Statistiche Detection per questo video
        tp_count = 0
        fp_count = 0
        fn_count = 0

        # Per Association: lista di tuple (gt_id, pred_id) per ogni match TP trovato
        matches_list = []

        for frame_id in all_frames:
            gts = gt_data.get(frame_id, [])
            preds = pred_data.get(frame_id, [])

            # Matrice IoU: righe=GT, colonne=Pred
            iou_matrix = np.zeros((len(gts), len(preds)))
            for i, gt in enumerate(gts):
                for j, pred in enumerate(preds):
                    iou_matrix[i, j] = self._calculate_iou(gt['box'], pred['box'])

            # Hungarian Algorithm (Maximizing IoU -> Minimizing 1-IoU)
            # Costruiamo matrice costi solo dove IoU >= alpha
            if iou_matrix.size > 0:
                cost_matrix = 1 - iou_matrix
                # Imposta costo infinito dove IoU < alpha per vietare l'associazione
                cost_matrix[iou_matrix < self.alpha] = 1000.0

                gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

                # Filtra match validi (IoU >= alpha e costo non proibitivo)
                valid_matches = []
                for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                    if iou_matrix[gt_idx, pred_idx] >= self.alpha:
                        valid_matches.append((gt_idx, pred_idx))

                        # Salva gli ID globali per il calcolo AssA
                        gt_id_global = gts[gt_idx]['obj_id']
                        pred_id_global = preds[pred_idx]['obj_id']
                        matches_list.append((gt_id_global, pred_id_global))

                num_matches = len(valid_matches)
                tp_count += num_matches
                fn_count += len(gts) - num_matches
                fp_count += len(preds) - num_matches

            else:
                # Caso base: o niente GT o niente Pred
                fn_count += len(gts)
                fp_count += len(preds)

        # --- CALCOLO ASSOCIATION SCORE (AssA) ---
        # Basato sulle definizioni TPA, FNA, FPA del PDF

        # 1. Conta quante volte ogni coppia (gt_id, pred_id) è stata matchata (TPA per quel match)
        match_counts = defaultdict(int)
        for m in matches_list:
            match_counts[m] += 1

        # 2. Conta quante volte ogni gt_id appare nei match totali
        gt_id_counts = defaultdict(int)
        # 3. Conta quante volte ogni pred_id appare nei match totali
        pred_id_counts = defaultdict(int)

        for gt_id, pred_id in matches_list:
            gt_id_counts[gt_id] += 1
            pred_id_counts[pred_id] += 1

        # 4. Calcola AssA sommando A(p) per ogni TP
        # A(p) = |TPA| / (|TPA| + |FNA| + |FPA|)
        assa_sum = 0.0

        for gt_id, pred_id in matches_list:
            tpa = match_counts[(gt_id, pred_id)]

            # FNA: GT è presente ma associato ad altro (o non associato in questo frame?
            # HOTA considera solo l'insieme dei TP per AssA).
            # Matematicamente: Tutte le volte che GT è stato matchato - le volte che è stato matchato con QUESTO pred
            fna = gt_id_counts[gt_id] - tpa

            # FPA: Pred è presente ma associato ad altro
            fpa = pred_id_counts[pred_id] - tpa

            # Formula pag 6
            a_p = tpa / (tpa + fna + fpa)
            assa_sum += a_p

        return {
            'TP': tp_count,
            'FP': fp_count,
            'FN': fn_count,
            'AssA_sum': assa_sum
        }

    @staticmethod
    def _calc_deta(tp, fn, fp):
        """Calcola DetA secondo formula """
        den = tp + fn + fp
        if den == 0: return 0.0
        return tp / den

    @staticmethod
    def _calc_assa(assa_sum, tp_count):
        """Calcola AssA come media delle A(p) sui TP """
        if tp_count == 0: return 0.0
        return assa_sum / tp_count

    def _load_data(self, file_path):
        """
        Legge file CSV formato MOT/SoccerNet.
        Ritorna dict: {frame_id: [{'obj_id': int, 'box': [x,y,w,h]}, ...]}
        """
        data = defaultdict(list)
        try:
            # Usa pandas per velocità, ignora colonne extra se presenti
            df = pd.read_csv(file_path, header=None)
            # Filtra solo le prime 6 colonne rilevanti
            # frame, id, x, y, w, h
            vals = df.iloc[:, :6].values

            for row in vals:
                frame_id = int(row[0])
                obj_id = int(row[1])
                x, y, w, h = row[2], row[3], row[4], row[5]

                # Ignora ID -1 se presenti (spesso usati per ignore regions)
                if obj_id == -1: continue

                # IMPORTANTE: Filtro Palla (Class 0).
                # Se i file GT contengono la palla, potremmo volerla escludere
                # dato che il PDF parla di "Player tracking".
                # Qui assumiamo che il GT sia pulito o che valutiamo tutto ciò che è nel file.

                data[frame_id].append({
                    'obj_id': obj_id,
                    'box': [x, y, w, h]
                })
        except Exception as e:
            # File vuoto o malformato
            pass
        return data

    @staticmethod
    def _calculate_iou(box1, box2):
        """Calcola Intersection over Union tra due box [x, y, w, h]"""
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
