import os
import csv
import sys
import traceback
from ultralytics import YOLO
import yaml

class Validator:
    """
    Gestisce il processo di validazione e grid search per modelli YOLO,
    permettendo di testare diverse soglie di confidenza e IoU per ottimizzare le performance.
    """

    def __init__(self, config, conf_mode):
        """
        Inizializza il validatore caricando il modello YOLO, configurando i percorsi dei dati
        e verificando l'integrità dei parametri forniti.
        """
        self.cfg = config
        self.cfg_values = conf_mode

        print(f"\n[Validator] {'=' * 50}")
        print("[Validator] INIZIALIZZAZIONE PIPELINE DI VALIDAZIONE")
        print(f"[Validator] {'=' * 50}")

        self._check_keys()

        weights_path = self.cfg['paths']['model_weights']
        print(f"[Validator] [INIT] Caricamento modello YOLO...")
        print(f"[Validator]        Path: {os.path.basename(weights_path)}")
        self.model = YOLO(weights_path)

        self.dataset_yaml = os.path.join(self.cfg['paths']['yolo_dataset'], 'dataset.yaml')
        self.conf_values = self.cfg_values['conf_threshold']
        self.iou_values = self.cfg_values['iou_threshold']

        with open(self.dataset_yaml, 'r') as f:
            d = yaml.safe_load(f)
            self.model.model.names = d['names']

        print(f"[Validator] [INIT] Dataset YAML: {os.path.basename(self.dataset_yaml)}")

    def _check_keys(self):
        """
        Valida la presenza di tutte le chiavi necessarie nel dizionario di configurazione
        prima di procedere con l'esecuzione.
        """
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
            print(f"[Validator] [ERROR] Configurazione incompleta. Chiavi mancanti: {list(missing_keys)}")
            sys.exit(1)

    @staticmethod
    def _calculate_deta(precision, recall):
        """
        Calcola la metrica DetA (Detection Accuracy) come combinazione armonica
        di precisione e recall.
        """
        if precision + recall == 0:
            return 0
        deta = (precision * recall) / (precision + recall - (precision * recall))
        return deta

    def run(self):
        """
        Esegue il loop di validazione (Grid Search) sui parametri IoU specificati,
        salva i risultati in formato CSV e identifica la configurazione migliore.
        """
        if self.cfg_values['verbose']:
            print(f"\n[Validator] {'=' * 50}")
            print(f"[Validator] AVVIO: GRID SEARCH SU MODELLO")
            print(f"[Validator] {'=' * 50}")

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
                    print(
                        f"\n[Validator] [Test {current_iter}/{total_iterations}] Conf={self.conf_values}, IoU={iou}...")

                try:
                    results = self.model.val(
                        data=self.dataset_yaml,
                        split=self.cfg['paths']['split'],
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
                        print(f"[Validator]     -> P={p:.4f}, R={r:.4f}, DetA={deta:.4f}")

                    writer.writerow([self.conf_values, iou, p, r, map50, deta])

                    file.flush()

                    if deta > best_deta:
                        best_deta = deta
                        best_params = {'conf': self.conf_values, 'iou': iou}
                        if self.cfg_values['verbose']:
                            print(f"[Validator]     >>> NUOVO RECORD! DetA: {deta:.4f}")

                except Exception as e:
                    print(f"[Validator] [ERROR] CRITICO (Conf={self.conf_values}, IoU={iou}): {e}")
                    traceback.print_exc()

        if self.cfg_values['verbose']:
            print(f"\n[Validator] {'=' * 50}")
            print(f"[Validator] [✔] GRID SEARCH COMPLETATA")
            print(f"[Validator] {'=' * 50}")
            print(f"[Validator]     Miglior DetA     : {best_deta:.4f}")
            print(f"[Validator]     Parametri        : {best_params}")
            print(f"[Validator]     File risultati   : {os.path.abspath(output_file)}")