import os
import shutil
import pandas as pd
from .common.io_manager import ReportManager
from .evaluation_helper import compute_nmae_from_behavior_files,build_trackeval_structure,compute_metrics_with_details


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

        print("\n" + "=" * 60)
        print("AVVIO VALUTAZIONE UFFICIALE: BEHAVIOR ANALYSIS (nMAE)")
        print("=" * 60)

        # Iteriamo sulle sottocartelle (es. test/behavior)
        for subdir in self.subdirs:
            print(f"\n--- Elaborazione Behavior directory: {subdir} ---")

            # Directory dove si trovano i tuoi file .txt generati
            pred_dir = os.path.join(self.base_output_dir, subdir)
            if not os.path.exists(pred_dir):
                print(f"Cartella predizioni non trovata: {pred_dir}")
                continue

            reporter = ReportManager(pred_dir)

            # --- CHIAMATA AL CODICE UFFICIALE ---
            # Le funzioni ufficiali richiedono stringhe per group (team_name)
            group_str = str(self.team_name)

            try:
                # La funzione ufficiale vuole: dataset_root, predictions_root, group
                # dataset_root è la cartella che contiene le cartelle video (es. test_set/videos)
                results = compute_nmae_from_behavior_files(
                    dataset_root=self.gt_base_dir,
                    predictions_root=pred_dir,
                    group=group_str
                )

                mae = results.get("MAE")
                nmae = results.get("nMAE")
                has_behavior = results.get("has_behavior", False)

                if has_behavior and nmae is not None:
                    print(f"[{subdir}] OFFICIAL RESULTS -> MAE: {mae:.4f} | nMAE: {nmae:.4f}")

                    # --- AGGIORNAMENTO REPORT (Mantiene compatibilità con il tuo sistema) ---
                    # Salviamo lo score globale nel JSON "execution_report.json"
                    report_update = {
                        "OFFICIAL_SCORES": {
                            "MAE": round(mae, 6),
                            "nMAE": round(nmae, 6)
                        }
                    }
                    reporter.update_json_section("evaluation_results", report_update)
                    print(f"Report JSON aggiornato: {reporter.json_path}")

                    # (Opzionale) Se vuoi salvare anche un CSV riassuntivo
                    csv_data = [{'Metric': 'nMAE', 'Score': nmae}, {'Metric': 'MAE', 'Score': mae}]
                    pd.DataFrame(csv_data).to_csv(os.path.join(pred_dir, 'official_behavior_metrics.csv'), index=False)
                else:
                    print(
                        f"[{subdir}] Impossibile calcolare nMAE (File mancanti o non validi secondo il tool ufficiale).")

            except Exception as e:
                print(f"Errore critico durante la valutazione ufficiale Behavior: {e}")

    def run_hota(self):
        """
        Wrapper Ufficiale TrackEval che salva anche il CSV dettagliato.
        """

        print("\n" + "=" * 60)
        print("AVVIO VALUTAZIONE UFFICIALE: TRACKING (HOTA + DETTAGLI)")
        print("=" * 60)

        for subdir in self.subdirs:
            print(f"\n--- Elaborazione HOTA directory: {subdir} ---")

            pred_dir = os.path.join(self.base_output_dir, subdir)
            if not os.path.exists(pred_dir):
                print(f"Cartella predizioni non trovata: {pred_dir}")
                continue

            reporter = ReportManager(pred_dir)
            group_str = str(self.team_name)
            fps = float(self.cfg.get('fps', 25.0))
            split_name = self.cfg['paths'].get('split_name', 'test')

            try:
                # 1. Costruzione Struttura (incluso il fix per skipping video mancanti)
                print("Costruzione struttura temporanea TrackEval...")
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

                # 2. Calcolo Metriche Dettagliate
                print("Esecuzione TrackEval e estrazione metriche...")
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
                    df = pd.DataFrame(detailed_results)

                    # Riorganizziamo le colonne per leggibilità
                    cols = ['Video', 'HOTA', 'DetA', 'AssA', 'MOTA', 'TP', 'FN', 'FP']
                    # Filtra solo colonne esistenti nel caso mancasse qualcosa
                    cols = [c for c in cols if c in df.columns]
                    df = df[cols]

                    csv_path = os.path.join(pred_dir, 'hota_metrics_official.csv')
                    df.to_csv(csv_path, index=False)
                    print(f"Report CSV dettagliato salvato: {csv_path}")

                    # Estraiamo HOTA globale per il log JSON
                    global_row = df[df['Video'] == 'GLOBAL_SCORE']
                    if not global_row.empty:
                        final_hota = global_row.iloc[0]['HOTA']
                        print("-" * 60)
                        print(f"[{subdir}] GLOBAL OFFICIAL HOTA: {final_hota:.6f}")
                        print("-" * 60)

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

                # 4. Pulizia
                if os.path.exists(self.trackeval_tmp_root):
                    shutil.rmtree(self.trackeval_tmp_root)

            except Exception as e:
                print(f"Errore critico durante la valutazione HOTA: {e}")
                import traceback
                traceback.print_exc()