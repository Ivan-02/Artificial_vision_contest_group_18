import os, shutil, glob
from pathlib import Path
import cv2
import trackeval
from typing import Dict, Tuple, List
import numpy as np

def natural_key(path: str) -> int:
    """Extracts numeric frame index from a file path for natural sorting."""
    name = os.path.basename(path)
    return int(name.split(".")[0])

def _read_behavior(path: str) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {} # (frame, region_id) -> n_people
    import csv
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            fr = int(row[0])
            rid = int(row[1])
            n = int(float(row[2]))
            out[(fr, rid)] = n
    return out

def compute_nmae_from_behavior_files(dataset_root: str, predictions_root: str, group: str) -> dict:
    """
    Computes MAE and nMAE globally over all videos found in BOTH GT and Predictions.
    Skips videos that are missing in predictions without failing.
    """
    abs_err_sum = 0.0
    n = 0

    # Recuperiamo tutti i video ID dalla cartella GT
    video_ids = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and d.isdigit()]
    video_ids.sort(key=lambda s: int(s))

    processed_videos = 0

    for vid in video_ids:
        gt_path = os.path.join(dataset_root, vid, "gt", "behavior_gt.txt")
        pr_path = os.path.join(predictions_root, f"behavior_{vid}_{group}.txt")

        # --- MODIFICA FONDAMENTALE ---
        # Se manca il file di predizione (es. hai runnato solo il video 116), saltiamo questo video
        # invece di invalidare l'intero test con has_all=False.
        if not os.path.isfile(pr_path):
            continue

        if not os.path.isfile(gt_path):
            continue

        processed_videos += 1

        # map (frame, region_id) -> n_people
        gt_b = _read_behavior(gt_path)
        pr_b = _read_behavior(pr_path)

        # Evaluate only where GT has an entry
        for key, gt_val in gt_b.items():
            pred_val = pr_b.get(key, 0)
            abs_err_sum += abs(pred_val - gt_val)
            n += 1

    # Se non abbiamo processato nessun frame (n=0), allora restituiamo errore/vuoto
    if n == 0:
        print(f"ATTENZIONE: Nessun file behavior valido trovato in {predictions_root}")
        return {"has_behavior": False, "MAE": None, "nMAE": None}

    mae = abs_err_sum / n
    nmae = (10.0 - min(10.0, max(0.0, mae))) / 10.0

    return {"has_behavior": True, "MAE": mae, "nMAE": nmae}

# ============================================================
# Build TrackEval temp folders in MOTChallenge2DBox layout
# ============================================================


def ensure_10col_and_force_class1(src_txt: str, dst_txt: str) -> None:
    """
    Writes a MOT 10-column file:
      frame,id,x,y,w,h,conf,class,vis,unused
    Forces class=1 to be compatible with TrackEval's pedestrian-class evaluation.
    """
    Path(dst_txt).parent.mkdir(parents=True, exist_ok=True)
    out_lines: List[str] = []

    with open(src_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            frame = parts[0]
            tid   = parts[1]
            x, y, w, h = parts[2:6]

            conf = parts[6] if len(parts) >= 7 else "1"
            cls  = "1"  # force pedestrian
            vis  = parts[8] if len(parts) >= 9 else "-1"
            z    = parts[9] if len(parts) >= 10 else "-1"

            out_lines.append(",".join([frame, tid, x, y, w, h, conf, cls, vis, z]))

    with open(dst_txt, "w") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

def write_seqinfo_ini(seq_dir: str, seq_name: str, fps: float, img_w: int, img_h: int, seq_len: int) -> None:
    """
    Minimal seqinfo.ini for TrackEval MotChallenge2DBox, used also to compute SoccerNet Challenge metrics.
    it is used to give info about frame size, fps, length, etc. to TrackEval.
    it will write a seqinfo.ini file in seq_dir.
    """
    content = "\n".join([
        "[Sequence]",
        f"name={seq_name}",
        "imDir=img1",
        f"frameRate={int(round(fps))}",
        f"seqLength={int(seq_len)}",
        f"imWidth={int(img_w)}",
        f"imHeight={int(img_h)}",
        "imExt=.jpg",
        ""
    ])
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f:
        f.write(content)


def list_video_ids(dataset_root: str) -> List[str]:
    vids = []
    for name in os.listdir(dataset_root):
        p = os.path.join(dataset_root, name)
        if os.path.isdir(p) and name.isdigit():
            vids.append(name)
    return sorted(vids, key=lambda s: int(s))



# ============================================================
# TrackEval call: compute only HOTA@alpha=0.50
# ============================================================
def build_trackeval_structure(
        dataset_root: str,
        predictions_root: str,
        group: str,
        split: str,
        fps: float,
        tmp_root: str,
        benchmark: str = "SNMOT",
        tracker_name: str = "test",
) -> Tuple[str, str, str]:
    """
    Creates temp structure for TrackEval.
    MODIFIED: Skips videos that do not have a prediction file instead of raising Error.
    """
    tmp_root = os.path.abspath(tmp_root)
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)

    gt_folder = os.path.join(tmp_root, "gt")
    tr_folder = os.path.join(tmp_root, "trackers")
    sm_folder = os.path.join(tmp_root, "seqmaps")
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(tr_folder, exist_ok=True)
    os.makedirs(sm_folder, exist_ok=True)

    bench_split = f"{benchmark}-{split}"
    gt_bs = os.path.join(gt_folder, bench_split)
    tr_bs = os.path.join(tr_folder, bench_split, tracker_name, "data")
    os.makedirs(gt_bs, exist_ok=True)
    os.makedirs(tr_bs, exist_ok=True)

    # 1. Recuperiamo TUTTI i video disponibili nella GT
    seqs = list_video_ids(dataset_root)
    if not seqs:
        raise FileNotFoundError(f"No numeric video folders found in: {dataset_root}")

    # Lista per tenere traccia dei video effettivamente processati (GT + PRED esistenti)
    valid_seqs = []

    for seq in seqs:
        src_seq = os.path.join(dataset_root, seq)
        src_img1 = os.path.join(src_seq, "img1")
        src_gt = os.path.join(src_seq, "gt", "gt.txt")
        src_pred = os.path.join(predictions_root, f"tracking_{seq}_{group}.txt")

        # --- MODIFICA CHIAVE QUI ---
        # Invece di lanciare errore, controlliamo se il file predizione esiste.
        # Se non esiste (es. hai tracciato solo il video 116), saltiamo questo video.
        if not os.path.isfile(src_pred):
            # Opzionale: decommenta la riga sotto se vuoi vedere quali video vengono saltati
            # print(f"Skipping video {seq}: Prediction file not found.")
            continue

        # Se manca la GT invece è grave, lasciamo l'errore o saltiamo (qui lascio l'errore originale)
        if not os.path.isfile(src_gt):
            print(f"Warning: Missing GT for existing prediction {seq}. Skipping.")
            continue
        # ---------------------------

        frame_paths = sorted(glob.glob(os.path.join(src_img1, "*.jpg")), key=natural_key)
        if not frame_paths:
            # Se non ci sono immagini, saltiamo
            continue

        im0 = cv2.imread(frame_paths[0])
        if im0 is None:
            continue

        H, W = im0.shape[:2]
        seq_len = len(frame_paths)

        # Destination GT sequence folder
        dst_seq = os.path.join(gt_bs, seq)
        os.makedirs(dst_seq, exist_ok=True)
        os.makedirs(os.path.join(dst_seq, "gt"), exist_ok=True)
        os.makedirs(os.path.join(dst_seq, "img1"), exist_ok=True)

        write_seqinfo_ini(dst_seq, seq_name=seq, fps=fps, img_w=W, img_h=H, seq_len=seq_len)

        ensure_10col_and_force_class1(src_gt, os.path.join(dst_seq, "gt", "gt.txt"))
        ensure_10col_and_force_class1(src_pred, os.path.join(tr_bs, f"{seq}.txt"))

        # Aggiungiamo il video alla lista di quelli validi da scrivere nella seqmap
        valid_seqs.append(seq)

    if not valid_seqs:
        print("ATTENZIONE: Nessun video valido trovato (coppia GT+Predizione). HOTA sarà 0.")

    # Seqmap file: Scriviamo SOLO i video che abbiamo effettivamente trovato e copiato
    seqmap_file = os.path.join(sm_folder, f"{bench_split}.txt")
    with open(seqmap_file, "w") as f:
        f.write("name\n")
        for seq in valid_seqs:
            f.write(f"{seq}\n")

    return gt_folder, tr_folder, seqmap_file


def compute_hota_at_05_trackeval(
    gt_folder: str,
    trackers_folder: str,
    seqmap_file: str,
    split: str,
    benchmark: str = "SNMOT",
    tracker_name: str = "test",
) -> float:
    """
    Runs TrackEval MotChallenge2DBox with only HOTA metric family,
    then extracts HOTA at alpha=0.50 (not averaged over alphas).
    """
    # --- configs (no argparse; direct dicts) ---
    eval_config = trackeval.Evaluator.get_default_eval_config() # get default eval config, as done in SoccerNet Challenge
    eval_config["DISPLAY_LESS_PROGRESS"] = True # show less progress otherwise too verbose, set to False to see full progress

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config() # get default dataset config for MotChallenge2DBox, as done in SoccerNet Challenge
    dataset_config.update({
        "BENCHMARK": benchmark,
        "GT_FOLDER": gt_folder,
        "TRACKERS_FOLDER": trackers_folder,
        "TRACKERS_TO_EVAL": [tracker_name],
        "SPLIT_TO_EVAL": split,
        "SEQMAP_FILE": seqmap_file,
        "DO_PREPROC": False,          # matches SoccerNet wrapper style
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "eval_results",
    }) # update with our specific paths and settings

    metrics_config = {"METRICS": ["HOTA"]} # only HOTA metric needed for the Challenge

    evaluator = trackeval.Evaluator(eval_config) # create evaluator with eval config
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)] # create dataset list with our dataset config
    metrics_list = [trackeval.metrics.HOTA(metrics_config)] # create metrics list with HOTA metric

    output_res, _ = evaluator.evaluate(dataset_list, metrics_list) # run official evaluation 

    # Find alpha index for 0.50 from the metric itself
    hota_metric = trackeval.metrics.HOTA(metrics_config) # create a HOTA curve metrics for an array of alphas
    alphas = np.array(hota_metric.array_labels, dtype=float) # get array of alphas used in HOTA curve
    idx = int(np.where(np.isclose(alphas, 0.5))[0][0]) # find index of alpha=0.50 

    # SoccerNet-style extraction uses COMBINED_SEQ and class key "pedestrian"
    hota_curve = output_res["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]["HOTA"]["HOTA"] # from the output results, get HOTA curve for combined sequences and pedestrian class 
    return float(hota_curve[idx]) # return HOTA@0.50 value as float

def compute_metrics_with_details(
        gt_folder: str,
        trackers_folder: str,
        seqmap_file: str,
        split: str,
        benchmark: str = "SNMOT",
        tracker_name: str = "test",
) -> List[Dict]:
    """
    Runs TrackEval to compute HOTA and CLEAR metrics.
    Returns a list of dictionaries containing detailed metrics (TP, FP, FN, etc.)
    for each video and the GLOBAL combination.
    """
    # 1. Configurazione TrackEval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_RESULTS"] = False
    eval_config["PRINT_ONLY_COMBINED"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "BENCHMARK": benchmark,
        "GT_FOLDER": gt_folder,
        "TRACKERS_FOLDER": trackers_folder,
        "TRACKERS_TO_EVAL": [tracker_name],
        "SPLIT_TO_EVAL": split,
        "SEQMAP_FILE": seqmap_file,
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "eval_results",
    })

    # 2. Richiediamo SIA HOTA (per gli score) SIA CLEAR (per TP/FP/FN espliciti)
    metrics_config = {"METRICS": ["HOTA", "CLEAR"]}

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    # Istanziamo le metriche
    metrics_list = []
    for metric in metrics_config['METRICS']:
        if metric == "HOTA":
            metrics_list.append(trackeval.metrics.HOTA())
        elif metric == "CLEAR":
            metrics_list.append(trackeval.metrics.CLEAR())

    # 3. Esecuzione
    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)

    # 4. Estrazione Dati Dettagliati
    # TrackEval restituisce: output_res[Dataset][Tracker][Sequence][Class][MetricFamily]

    # Troviamo l'indice per alpha=0.5 (usato per HOTA standard)
    hota_metric = trackeval.metrics.HOTA()
    alphas = np.array(hota_metric.array_labels, dtype=float)
    idx_05 = int(np.where(np.isclose(alphas, 0.5))[0][0])

    detailed_results = []

    # Otteniamo i dati per il nostro tracker
    tracker_data = output_res["MotChallenge2DBox"][tracker_name]

    # Iteriamo su tutte le sequenze (video) + 'COMBINED_SEQ' (Globale)
    # L'ordine delle chiavi non è garantito, ma 'COMBINED_SEQ' è sempre presente
    for seq_key in tracker_data.keys():
        # Dati HOTA (Scores)
        hota_res = tracker_data[seq_key]["pedestrian"]["HOTA"]
        # Dati CLEAR (Conteggi TP/FP/FN)
        clear_res = tracker_data[seq_key]["pedestrian"]["CLEAR"]

        # Estrazione Valori
        row = {
            'Video': seq_key if seq_key != 'COMBINED_SEQ' else 'GLOBAL_SCORE',
            'HOTA': float(hota_res['HOTA'][idx_05]),
            'DetA': float(hota_res['DetA'][idx_05]),
            'AssA': float(hota_res['AssA'][idx_05]),
            'TP': int(clear_res['CLR_TP']),
            'FN': int(clear_res['CLR_FN']),
            'FP': int(clear_res['CLR_FP']),
        }
        detailed_results.append(row)

    def sort_key(x):
        if x['Video'] == 'GLOBAL_SCORE': return 999999
        try:
            return int(x['Video'])
        except:
            return 0

    detailed_results.sort(key=sort_key)

    return detailed_results