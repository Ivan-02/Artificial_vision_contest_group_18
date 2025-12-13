import os
from ultralytics import YOLO


class Validator:
    def __init__(self, config):
        self.cfg = config
        # Carica il modello (può essere quello pre-addestrato o uno trainato da te)
        self.model = YOLO(self.cfg['paths']['model_weights'])

    def validate(self):
        """
        Esegue la validazione del modello calcolando mAP, Precision e Recall.
        Richiede che il dataset sia stato convertito in formato YOLO (images/labels + dataset.yaml).
        """
        print(f"--- Avvio Validazione con modello: {self.cfg['paths']['model_weights']} ---")

        # Il percorso al file dataset.yaml creato dal DataManager
        # Assumiamo sia in dataset/yolo/dataset.yaml
        dataset_yaml_path = os.path.join(self.cfg['paths']['yolo_dataset'], 'dataset.yaml')

        if not os.path.exists(dataset_yaml_path):
            print(f"ERRORE: Non trovo il file {dataset_yaml_path}")
            print("Hai eseguito 'python main.py --mode prepare' prima?")
            return

        # Esegui model.val()
        metrics = self.model.val(
            data=dataset_yaml_path,  # Il file che descrive dove sono immagini e labels
            split=self.cfg['val']['split'],  # 'val' o 'test'
            imgsz=self.cfg['val']['imgsz'],
            batch=self.cfg['val']['batch_size'],
            conf=self.cfg['val']['conf_threshold'],
            iou=self.cfg['val']['iou_threshold'],
            project=self.cfg['paths']['output_val'],
            name="validation_results",
            device=self.cfg['training']['device'],
            plots=True  # Genera grafici (matrice confusione, curve P-R)
        )

        # Stampa un riassunto rapido
        print("\n--- Risultati Validazione ---")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print(f"Grafici salvati in: {self.cfg['paths']['output_val']}/validation_results")