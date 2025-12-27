import os
import json

class ReportManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.json_path = os.path.join(self.output_dir, "execution_report.json")

    def save_txt_results(self, filename, lines, append=False):
        mode = 'a' if append else 'w'
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, mode) as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Errore scrittura {filename}: {e}")

    def load_json_report(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def update_json_section(self, section_key, data_dict):
        current_data = self.load_json_report()

        if section_key not in current_data:
            current_data[section_key] = {}

        current_data[section_key].update(data_dict)

        try:
            with open(self.json_path, 'w') as f:
                json.dump(current_data, f, indent=4)
        except Exception as e:
            print(f"Errore salvataggio JSON: {e}")