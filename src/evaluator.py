import os
import shutil
import pandas as pd
import traceback
from .common.io_manager import ReportManager
from .evaluation_helper import compute_nmae_from_behavior_files, build_trackeval_structure, compute_metrics_with_details


class Evaluator:
    def __init__(self, config):
        self.cfg = config
        self.base_output_dir = self.cfg['paths']['output_submission']
        self.subdirs = self.cfg['paths']['output_subdirs']
        self.gt_base_dir = os.path.join(self.cfg['paths']['raw_data'], self.cfg['paths']['split'])
        self.team_name = self.cfg['names']['team']  # Es. '18' o 18
        self.trackeval_tmp_root = os.path.join(self.base_output_dir, "_tmp_trackeval_wrapper")

    def run_behavior(self):
        """
        Wrapper per il calcolo nMAE Ufficiale.
        Mantiene la firma originale ma usa la logica ufficiale internamente.
        """
        print(f"\n[Evaluator] {'=' * 50}")
        print("[Evaluator] AVVIO: BEHAVIOR ANALYSIS (nMAE)")
        print(f"[Evaluator] {'=' * 50}")

        # Iteriamo sulle sottocartelle (es. test/behavior)
        for subdir in self.subdirs:
            print(f"\n[Evaluator] >>> Elaborazione directory: {subdir}")

            # Directory dove si trovano i tuoi file .txt generati
            pred_dir = os.path.join(self.base_output_dir, subdir)
            if not os.path.exists(pred_dir):
                print(f"[Evaluator] [!] ATTENZIONE: Cartella predizioni non trovata -> {pred_dir}")
                continue

            reporter = ReportManager(pred_dir)

            # --- CHIAMATA AL CODICE UFFICIALE ---
            group_str = str(self.team_name)

            try:
                # La funzione ufficiale vuole: dataset_root, predictions_root, group
                results = compute_nmae_from_behavior_files(
                    dataset_root=self.gt_base_dir,
                    predictions_root=pred_dir,
                    group=group_str
                )

                mae = results.get("MAE")
                nmae = results.get("nMAE")
                has_behavior = results.get("has_behavior", False)

                if has_behavior and nmae is not None:
                    print(f"[Evaluator]     Risultati Ufficiali [{subdir}]:")
                    print(f"[Evaluator]     -------------------------------")
                    print(f"[Evaluator]     MAE  : {mae:.4f}")
                    print(f"[Evaluator]     nMAE : {nmae:.4f}")
                    print(f"[Evaluator]     -------------------------------")

                    # --- AGGIORNAMENTO REPORT ---
                    report_update = {
                        "OFFICIAL_SCORES": {
                            "MAE": round(mae, 6),
                            "nMAE": round(nmae, 6)
                        }
                    }
                    reporter.update_json_section("evaluation_results", report_update)
                    print(f"[Evaluator] [✔] Report JSON aggiornato: {os.path.basename(reporter.json_path)}")

                    # (Opzionale) Se vuoi salvare anche un CSV riassuntivo
                    csv_data = [{'Metric': 'nMAE', 'Score': nmae}, {'Metric': 'MAE', 'Score': mae}]
                    pd.DataFrame(csv_data).to_csv(os.path.join(pred_dir, 'official_behavior_metrics.csv'), index=False)
                else:
                    print(f"[Evaluator] [!] Impossibile calcolare nMAE per {subdir} (File mancanti o non validi).")

            except Exception as e:
                print(f"[Evaluator] [ERROR] Errore critico Behavior Analysis: {e}")

    def run_hota(self):
        """
        Wrapper Ufficiale TrackEval che salva anche il CSV dettagliato.
        """
        print(f"\n[Evaluator] {'=' * 50}")
        print("[Evaluator] AVVIO: TRACKING ANALYSIS (HOTA)")
        print(f"[Evaluator] {'=' * 50}")

        for subdir in self.subdirs:
            print(f"\n[Evaluator] >>> Elaborazione directory: {subdir}")

            pred_dir = os.path.join(self.base_output_dir, subdir)
            if not os.path.exists(pred_dir):
                print(f"[Evaluator] [!] ATTENZIONE: Cartella predizioni non trovata -> {pred_dir}")
                continue

            reporter = ReportManager(pred_dir)
            group_str = str(self.team_name)
            fps = float(self.cfg.get('fps', 25.0))
            split_name = self.cfg['paths'].get('split_name', 'test')

            try:
                # 1. Costruzione Struttura
                print("[Evaluator]     [1/3] Costruzione struttura temporanea TrackEval...")
                gt_folder, tr_folder, seqmap_file = build_trackeval_structure(
                    dataset_root=self.gt_base_dir,
                    predictions_root=pred_dir,
                    group=group_str,
                    split=split_name,
                    fps=fps,
                    tmp_root=self.trackeval_tmp_root,
                    benchmark="SNMOT",
                    tracker_name="test"
                )

                # 2. Calcolo Metriche
                print("[Evaluator]     [2/3] Esecuzione TrackEval e calcolo metriche...")
                detailed_results = compute_metrics_with_details(
                    gt_folder=gt_folder,
                    trackers_folder=tr_folder,
                    seqmap_file=seqmap_file,
                    split=split_name,
                    benchmark="SNMOT",
                    tracker_name="test"
                )

                # 3. Creazione e Salvataggio CSV
                if detailed_results:
                    print("[Evaluator]     [3/3] Elaborazione risultati finali...")
                    df = pd.DataFrame(detailed_results)

                    # Riorganizziamo le colonne per leggibilità
                    cols = ['Video', 'HOTA', 'DetA', 'AssA', 'MOTA', 'TP', 'FN', 'FP']
                    cols = [c for c in cols if c in df.columns]
                    df = df[cols]

                    csv_path = os.path.join(pred_dir, 'hota_metrics_official.csv')
                    df.to_csv(csv_path, index=False)
                    print(f"[Evaluator] [✔] CSV dettagliato salvato: {os.path.basename(csv_path)}")

                    # Estraiamo HOTA globale per il log JSON
                    global_row = df[df['Video'] == 'GLOBAL_SCORE']
                    if not global_row.empty:
                        final_hota = global_row.iloc[0]['HOTA']

                        print(f"[Evaluator]     -------------------------------")
                        print(f"[Evaluator]     GLOBAL HOTA SCORE: {final_hota:.6f}")
                        print(f"[Evaluator]     -------------------------------")

                        # Aggiornamento JSON
                        report_update = {
                            "OFFICIAL_SCORES": {
                                "HOTA_05": round(final_hota, 6),
                                "TP_total": int(global_row.iloc[0]['TP']),
                                "FP_total": int(global_row.iloc[0]['FP']),
                                "FN_total": int(global_row.iloc[0]['FN'])
                            }
                        }
                        reporter.update_json_section("evaluation_results", report_update)
                        print(f"[Evaluator] [✔] Report JSON aggiornato con HOTA Score.")

                else:
                    print("[Evaluator] [!] ATTENZIONE: Nessun risultato calcolato.")
                    print("[Evaluator]     Possibili cause: Nessun video in comune tra GT e Predizioni, o file vuoti.")

                if os.path.exists(self.trackeval_tmp_root):
                    shutil.rmtree(self.trackeval_tmp_root)

            except Exception as e:
                print(f"[Evaluator] [ERROR] Errore critico Tracking Analysis: {e}")
                traceback.print_exc()