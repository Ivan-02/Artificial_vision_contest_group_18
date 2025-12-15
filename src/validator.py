import os
import csv
import sys
import traceback
from ultralytics import YOLO
import yaml

class Validator:
    def __init__(self, config, conf_mode):
        self.cfg = config
        self.cfg_values = conf_mode
        self._check_keys()
        self.model = YOLO(self.cfg['paths']['model_weights'])
        self.dataset_yaml = os.path.join(self.cfg['paths']['yolo_dataset'], 'dataset.yaml')
        self.conf_values = self.cfg_values['val']['conf_threshold']
        self.iou_values = self.cfg_values['val']['iou_threshold']

        with open(os.path.join(self.cfg['paths']['yolo_dataset'], 'dataset.yaml'), 'r') as f:
            d = yaml.safe_load(f)
            self.model.model.names = d['names']


    def _check_keys(self):
        required_keys = {
            'verbose',
            'imgsz',
            'classes',
            'batch_size',
            'conf_threshold',
            'single_cls',
            'iou_threshold',
            'plots',
            'exist_ok'
        }

        current_keys = set(self.cfg_values.keys())
        missing_keys = required_keys - current_keys

        if missing_keys:
            print(f"[ERRORE] Configurazione incompleta. Chiavi mancanti: {list(missing_keys)}")
            sys.exit(1)

    @staticmethod
    def _calculate_deta(precision, recall):
        if precision + recall == 0:
            return 0
        deta = (precision * recall) / (precision + recall - (precision * recall))
        return deta

    def run(self):
        if self.cfg_values['verbose']:
            print(f"--- Avvio Grid Search su modello: {self.cfg['paths']['model_weights']}")

        output_file = os.path.join(self.cfg['paths']['output_val'], 'grid_search_results.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        best_deta = 0
        best_params = {}

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Conf', 'IoU', 'Precision', 'Recall', 'mAP50', 'DetA_Score'])

            total_iterations = len(self.iou_values)
            current_iter = 0

            for iou in self.iou_values:
                current_iter += 1

                if self.cfg_values['verbose']:
                    print(f"\n[Test {current_iter}/{total_iterations}] Conf={self.conf_values}, IoU={iou}...")

                try:
                    results = self.model.val(
                        data=self.dataset_yaml,
                        split=self.cfg['path']['split'],
                        classes=self.cfg_values['classes'],
                        single_cls=self.cfg_values['single_cls'],
                        imgsz=self.cfg_values['imgsz'],
                        batch=self.cfg_values['batch_size'],
                        project=self.cfg['paths']['output_val'],
                        name=f"gs_{self.conf_values}_{iou}",
                        conf=self.conf_values,
                        iou=iou,
                        device=self.cfg['device'],
                        plots=self.cfg_values['plots'],
                        verbose=self.cfg_values['verbose'],
                        exist_ok=self.cfg_values['exist_ok'],
                    )

                    p = results.box.mp
                    r = results.box.mr
                    map50 = results.box.map50

                    deta = self._calculate_deta(p, r)

                    if self.cfg_values['verbose']:
                        print(f"   -> P={p:.4f}, R={r:.4f}, DetA={deta:.4f}")

                    writer.writerow([self.conf_values, iou, p, r, map50, deta])

                    file.flush()

                    if deta > best_deta:
                        best_deta = deta
                        best_params = {'conf': self.conf_values, 'iou': iou}
                        if self.cfg_values['verbose']:
                            print(f"   >>> NUOVO RECORD! DetA: {deta:.4f}")

                except Exception as e:
                    print(f"ERRORE CRITICO (Conf={self.conf_values}, IoU={iou}):")
                    traceback.print_exc()

        if self.cfg_values['verbose']:
            print("\n" + "=" * 40)
            print(f"Miglior DetA: {best_deta:.4f}")
            print(f"Parametri Vincenti: {best_params}")
            print(f"File risultati: {output_file}")
            print("=" * 40)