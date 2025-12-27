import os
import json

class ReportManager:
    """
    Gestisce la creazione e l'aggiornamento dei file di report (TXT e JSON)
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