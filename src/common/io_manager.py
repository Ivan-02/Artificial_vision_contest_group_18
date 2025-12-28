import os
import json
from tqdm import tqdm


class ReportManager:
    """
    Gestisce la creazione e l'aggiornamento e il caricamento dei file di report e
    di configurazione (TXT e JSON)
    e assicura la gestione corretta della directory di output.
    """

    def __init__(self, output_dir):
        """
        Inizializza il gestore, crea la directory di output se non esiste
        e definisce il percorso per il report cumulativo JSON.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.json_path = os.path.join(self.output_dir, "execution_report.json")

    def load_rois(self, video_path, global_roi_path=None):
        """
        Carica le ROI seguendo una logica a priorità:
        1. Cerca 'roi.json' specifico all'interno della cartella del video.
        2. Se non presente, cerca la definizione nel file ROI globale (da config).
        3. Se fallisce anche il globale, usa un fallback hardcoded.
        """

        fallback = {
            "roi1": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.4},
            "roi2": {"x": 0.5, "y": 0.7, "width": 0.5, "height": 0.3}
        }

        local_roi_path = os.path.join(video_path, 'roi.json')

        if os.path.exists(local_roi_path):
            try:
                with open(local_roi_path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                tqdm.write(f"[BehaviorAnalyzer] [WARN] File ROI locale trovato ma corrotto per {video_path}: {e}")


        if global_roi_path and os.path.exists(global_roi_path):
            try:
                with open(global_roi_path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                tqdm.write(f"[BehaviorAnalyzer] [WARN] Errore lettura file ROI globale: {e}")

        tqdm.write(f"[BehaviorAnalyzer] [WARN] Nessuna ROI trovata per {video_path}. Uso Fallback.")
        return fallback

    def save_txt_results(self, filename, lines, append=False):
        """
        Scrive o accoda una lista di stringhe su un file di testo specifico,
        gestendo le modalità di apertura ('w' o 'a') e catturando errori di I/O.
        """
        mode = 'a' if append else 'w'
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, mode) as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Errore scrittura {filename}: {e}")

    def load_json_report(self):
        """
        Tenta di caricare il report JSON esistente; restituisce un dizionario vuoto
        se il file non esiste o si verificano errori di lettura.
        """
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def update_json_section(self, section_key, data_dict):
        """
        Aggiorna una specifica sezione del report JSON con nuovi dati,
        preservando i dati esistenti e salvando il file con indentazione.
        """
        current_data = self.load_json_report()

        if section_key not in current_data:
            current_data[section_key] = {}

        current_data[section_key].update(data_dict)

        try:
            with open(self.json_path, 'w') as f:
                json.dump(current_data, f, indent=4)
        except Exception as e:
            print(f"Errore salvataggio JSON: {e}")