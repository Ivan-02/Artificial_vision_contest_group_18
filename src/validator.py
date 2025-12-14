import os
import csv
import traceback
from ultralytics import YOLO


class Validator:
    def __init__(self, config, conf_values=None, iou_values=None):
        self.cfg = config
        self.model = YOLO(self.cfg['paths']['model_weights'])
        self.dataset_yaml = os.path.join(self.cfg['paths']['yolo_dataset'], 'dataset.yaml')

        # Se non passiamo valori, usiamo quelli di default
        if conf_values is None:
            self.conf_values = [0.20, 0.25, 0.35]
        else:
            self.conf_values = conf_values

        if iou_values is None:
            self.iou_values = [0.50, 0.55, 0.65]
        else:
            self.iou_values = iou_values

    @staticmethod
    def _calculate_deta(precision, recall):
        if precision + recall == 0:
            return 0
        deta = (precision * recall) / (precision + recall - (precision * recall))
        return deta

    def run(self, verbose=False):
        # 1. FIX SINTASSI: Uso apici singoli dentro l'f-string
        if verbose:
            print(f"--- Avvio Grid Search su modello: {self.cfg['paths']['model_weights']}")

        output_file = os.path.join(self.cfg['paths']['output_val'], 'grid_search_results.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        best_deta = 0
        best_params = {}

        # Recupera le classi dal config (es. [1, 2, 3])
        # Questo è il filtro magico che ignorerà la palla (classe 0)
        target_classes = self.cfg['val'].get('classes', [1, 2, 3])

        # Apertura CSV con buffer azzerato
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Conf', 'IoU', 'Precision', 'Recall', 'mAP50', 'DetA_Score'])

            total_iterations = len(self.conf_values) * len(self.iou_values)
            current_iter = 0

            for conf in self.conf_values:
                for iou in self.iou_values:
                    current_iter += 1

                    if verbose:
                        print(f"\n[Test {current_iter}/{total_iterations}] Conf={conf}, IoU={iou}...")

                    try:
                        # ESEGUE LA VALIDAZIONE
                        results = self.model.val(
                            data=self.dataset_yaml,
                            split=self.cfg['val']['split'],

                            # A. FILTRO CLASSI:
                            # Passando [1, 2, 3], YOLO caricherà le label ma scarterà
                            # tutte quelle con ID 0 (la palla).
                            classes=target_classes,

                            # B. UNIFICAZIONE:
                            # Tratta 1, 2, 3 come se fossero tutti la stessa classe ("persona").
                            # Fondamentale per il tuo obiettivo.
                            single_cls=True,

                            imgsz=self.cfg['val']['imgsz'],
                            batch=self.cfg['val']['batch_size'],
                            project=self.cfg['paths']['output_val'],
                            name=f"gs_{conf}_{iou}",
                            conf=conf,
                            iou=iou,
                            device=self.cfg['training']['device'],
                            plots=True,  # Metti a True se vuoi vedere le immagini di debug
                            verbose=False,
                            exist_ok=True
                        )

                        # Estrazione metriche
                        p = results.box.mp
                        r = results.box.mr
                        map50 = results.box.map50

                        deta = self._calculate_deta(p, r)

                        if verbose:
                            print(f"   -> P={p:.4f}, R={r:.4f}, DetA={deta:.4f}")

                        # Scrittura su CSV
                        writer.writerow([conf, iou, p, r, map50, deta])

                        # 2. FIX BUFFER: Forza la scrittura immediata su disco
                        file.flush()

                        if deta > best_deta:
                            best_deta = deta
                            best_params = {'conf': conf, 'iou': iou}
                            if verbose:
                                print(f"   >>> NUOVO RECORD! DetA: {deta:.4f}")

                    except Exception as e:
                        print(f"ERRORE CRITICO (Conf={conf}, IoU={iou}):")
                        # 3. FIX DEBUG: Stampa l'errore completo
                        traceback.print_exc()

        if verbose:
            print("\n" + "=" * 40)
            print(f"Miglior DetA: {best_deta:.4f}")
            print(f"Parametri Vincenti: {best_params}")
            print(f"File risultati: {output_file}")
            print("=" * 40)