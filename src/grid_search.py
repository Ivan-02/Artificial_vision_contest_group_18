import os
import csv
from ultralytics import YOLO
import numpy as np


class GridSearch:
    def __init__(self, config):
        self.cfg = config
        self.model_path = self.cfg['paths']['model_weights']
        self.dataset_yaml = os.path.join(self.cfg['paths']['yolo_dataset'], 'dataset.yaml')

        # DEFINISCI QUI I RANGE DA TESTARE
        # Per SoccerNet consiglio questi valori:
        self.conf_values = [0.15, 0.30, 0.45]
        self.iou_values = [0.45, 0.60, 0.7]

    def calculate_deta(self, precision, recall):
        """
        Calcola DetA (Detection Accuracy) partendo da Precision e Recall.
        Formula derivata: DetA = TP / (TP + FN + FP)
        """
        if precision + recall == 0:
            return 0
        # Formula algebrica per ottenere DetA da P e R
        deta = (precision * recall) / (precision + recall - (precision * recall))
        return deta

    def run(self):
        print(f"--- Avvio Grid Search su modello: {self.model_path} ---")

        # Carica il modello una volta sola
        model = YOLO(self.model_path)

        # File di output per i risultati
        output_file = os.path.join(self.cfg['paths']['output_val'], 'grid_search_results.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        best_deta = 0
        best_params = {}

        # Apri il CSV per scrivere i risultati man mano
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Conf', 'IoU', 'Precision', 'Recall', 'mAP50', 'DetA_Score'])

            # Ciclo annidato (La Grid Search vera e propria)
            total_iterations = len(self.conf_values) * len(self.iou_values)
            current_iter = 0

            for conf in self.conf_values:
                for iou in self.iou_values:
                    current_iter += 1
                    print(f"\n[Test {current_iter}/{total_iterations}] Testing Conf={conf}, IoU={iou}...")

                    try:
                        # Esegui validazione silenziosa (verbose=False)
                        results = model.val(
                            data=self.dataset_yaml,
                            split=self.cfg['val']['split'],
                            imgsz=self.cfg['val']['imgsz'],
                            batch=self.cfg['val']['batch_size'],
                            conf=conf,
                            iou=iou,
                            device=self.cfg['training']['device'],
                            plots=False,
                            verbose=False
                        )

                        # Estrai metriche
                        # results.box.mp = Mean Precision
                        # results.box.mr = Mean Recall
                        # results.box.map50 = mAP a IoU 0.5
                        p = results.box.mp
                        r = results.box.mr
                        map50 = results.box.map50

                        # Calcola la TUA metrica custom
                        deta = self.calculate_deta(p, r)

                        # Stampa risultato parziale
                        print(f"   -> Result: P={p:.4f}, R={r:.4f}, DetA={deta:.4f}")

                        # Salva su CSV
                        writer.writerow([conf, iou, p, r, map50, deta])

                        # Aggiorna il migliore
                        if deta > best_deta:
                            best_deta = deta
                            best_params = {'conf': conf, 'iou': iou}
                            print(f"   >>> NUOVO RECORD! DetA: {deta:.4f}")

                    except Exception as e:
                        print(f"Errore durante iterazione conf={conf}, iou={iou}: {e}")

        print("\n" + "=" * 40)
        print("GRID SEARCH COMPLETATA")
        print("=" * 40)
        print(f"Miglior DetA: {best_deta:.4f}")
        print(f"Migliori Parametri: Conf={best_params.get('conf')}, IoU={best_params.get('iou')}")
        print(f"Risultati salvati in: {output_file}")
        print("=" * 40)