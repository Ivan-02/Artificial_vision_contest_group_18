import os
import shutil
import glob
import sys
from contextlib import contextmanager
from pathlib import Path
import cv2
import trackeval
from typing import Dict, Tuple, List
import numpy as np

"""
Modulo di utility per la valutazione delle performance: include funzioni per il calcolo del nMAE (comportamento), 
la formattazione dei dati in standard MOTChallenge e l'integrazione con il framework TrackEval per le metriche HOTA.
"""

@contextmanager
def suppress_stdout():
    """
    Gestore di contesto per silenziare temporaneamente l'output standard (stdout),
    utile per evitare log ridondanti da librerie esterne.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def natural_key(path: str) -> int:
    """
    Genera una chiave numerica a partire dal nome del file per consentire
    un ordinamento naturale (es. '2.jpg' prima di '10.jpg').
    """
    name = os.path.basename(path)
    try:
        return int(name.split(".")[0])
    except ValueError:
        return 0


def _read_behavior(path: str) -> Dict[Tuple[int, int], int]:
    """
    Legge un file di behavior e mappa le coppie (frame, id) al valore rilevato,
    gestendo eventuali errori di parsing.
    """
    out: Dict[Tuple[int, int], int] = {}
    import csv
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                fr = int(row[0])
                rid = int(row[1])
                n = int(float(row[2]))
                out[(fr, rid)] = n
    except Exception:
        return {}
    return out

def compute_nmae_from_behavior_files(dataset_root: str, predictions_root: str, group: str) -> dict:
    """
    Calcola l'errore medio assoluto (MAE) e lo normalizza (nMAE) confrontando
    le predizioni di comportamento con la Ground Truth su tutto il dataset.
    """
    abs_err_sum = 0.0
    n = 0

    if not os.path.exists(dataset_root):
        return {"has_behavior": False, "MAE": None, "nMAE": None}

    video_ids = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and d.isdigit()]
    video_ids.sort(key=lambda s: int(s))

    processed_videos = 0

    for vid in video_ids:
        gt_path = os.path.join(dataset_root, vid, "gt", "behavior_gt.txt")
        pr_path = os.path.join(predictions_root, f"behavior_{vid}_{group}.txt")

        if not os.path.isfile(pr_path):
            continue
        if not os.path.isfile(gt_path):
            continue

        processed_videos += 1

        gt_b = _read_behavior(gt_path)
        pr_b = _read_behavior(pr_path)

        for key, gt_val in gt_b.items():
            pred_val = pr_b.get(key, 0)
            abs_err_sum += abs(pred_val - gt_val)
            n += 1

    if n == 0:
        return {"has_behavior": False, "MAE": None, "nMAE": None}

    mae = abs_err_sum / n
    nmae = (10.0 - min(10.0, max(0.0, mae))) / 10.0

    return {"has_behavior": True, "MAE": mae, "nMAE": nmae}

def ensure_10col_and_force_class1(src_txt: str, dst_txt: str) -> None:
    """
    Normalizza i file di tracking convertendoli nel formato standard a 10 colonne
    e forzando l'ID di classe a 1 per la compatibilità con i benchmark.
    """
    Path(dst_txt).parent.mkdir(parents=True, exist_ok=True)
    out_lines: List[str] = []

    if not os.path.exists(src_txt):
        return

    with open(src_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            frame = parts[0]
            tid = parts[1]
            x, y, w, h = parts[2:6]
            conf = parts[6] if len(parts) >= 7 else "1"
            cls = "1"
            vis = parts[8] if len(parts) >= 9 else "-1"
            z = parts[9] if len(parts) >= 10 else "-1"

            out_lines.append(",".join([frame, tid, x, y, w, h, conf, cls, vis, z]))

    with open(dst_txt, "w") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))


def write_seqinfo_ini(seq_dir: str, seq_name: str, fps: float, img_w: int, img_h: int, seq_len: int) -> None:
    """
    Genera il file di configurazione 'seqinfo.ini' necessario per definire
    proprietà come FPS e risoluzione all'interno del framework TrackEval.
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
    """
    Scansiona la directory del dataset per identificare e restituire una lista
    ordinata di ID video validi (cartelle numeriche).
    """
    vids = []
    if not os.path.exists(dataset_root):
        return []
    for name in os.listdir(dataset_root):
        p = os.path.join(dataset_root, name)
        if os.path.isdir(p) and name.isdigit():
            vids.append(name)
    return sorted(vids, key=lambda s: int(s))


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
    Costruisce una gerarchia di cartelle temporanea compatibile con MOTChallenge,
    copiando e formattando GT e predizioni per l'analisi automatizzata.
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

    seqs = list_video_ids(dataset_root)
    valid_seqs = []

    for seq in seqs:
        src_seq = os.path.join(dataset_root, seq)
        src_img1 = os.path.join(src_seq, "img1")
        src_gt = os.path.join(src_seq, "gt", "gt.txt")
        src_pred = os.path.join(predictions_root, f"tracking_{seq}_{group}.txt")

        if not os.path.isfile(src_pred):
            continue

        if not os.path.isfile(src_gt):
            continue

        frame_paths = sorted(glob.glob(os.path.join(src_img1, "*.jpg")), key=natural_key)
        if not frame_paths:
            continue

        im0 = cv2.imread(frame_paths[0])
        if im0 is None:
            continue

        H, W = im0.shape[:2]
        seq_len = len(frame_paths)

        dst_seq = os.path.join(gt_bs, seq)
        os.makedirs(dst_seq, exist_ok=True)
        os.makedirs(os.path.join(dst_seq, "gt"), exist_ok=True)
        os.makedirs(os.path.join(dst_seq, "img1"), exist_ok=True)

        write_seqinfo_ini(dst_seq, seq_name=seq, fps=fps, img_w=W, img_h=H, seq_len=seq_len)

        ensure_10col_and_force_class1(src_gt, os.path.join(dst_seq, "gt", "gt.txt"))
        ensure_10col_and_force_class1(src_pred, os.path.join(tr_bs, f"{seq}.txt"))

        valid_seqs.append(seq)

    seqmap_file = os.path.join(sm_folder, f"{bench_split}.txt")
    with open(seqmap_file, "w") as f:
        f.write("name\n")
        for seq in valid_seqs:
            f.write(f"{seq}\n")

    return gt_folder, tr_folder, seqmap_file


def compute_metrics_with_details(
        gt_folder: str,
        trackers_folder: str,
        seqmap_file: str,
        split: str,
        benchmark: str = "SNMOT",
        tracker_name: str = "test",
) -> List[Dict]:
    """
    Interfaccia il motore TrackEval per calcolare metriche avanzate (HOTA, MOTA, DetA, AssA)
    e restituisce un dizionario dettagliato con i risultati per ogni video.
    """
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_RESULTS"] = False
    eval_config["PRINT_ONLY_COMBINED"] = False
    eval_config["PRINT_CONFIG"] = False

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

    metrics_config = {"METRICS": ["HOTA", "CLEAR"]}

    with suppress_stdout():
        try:
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

            metrics_list = []
            for metric in metrics_config['METRICS']:
                if metric == "HOTA":
                    metrics_list.append(trackeval.metrics.HOTA())
                elif metric == "CLEAR":
                    metrics_list.append(trackeval.metrics.CLEAR())

            output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
        except Exception:
            return []

    hota_metric = trackeval.metrics.HOTA()
    alphas = np.array(hota_metric.array_labels, dtype=float)

    if "MotChallenge2DBox" not in output_res:
        return []

    try:
        idx_05 = int(np.where(np.isclose(alphas, 0.5))[0][0])
        tracker_data = output_res["MotChallenge2DBox"][tracker_name]
    except (KeyError, IndexError):
        return []

    detailed_results = []

    for seq_key in tracker_data.keys():
        try:
            hota_res = tracker_data[seq_key]["pedestrian"]["HOTA"]
            clear_res = tracker_data[seq_key]["pedestrian"]["CLEAR"]

            row = {
                'Video': seq_key if seq_key != 'COMBINED_SEQ' else 'GLOBAL_SCORE',
                'HOTA': float(hota_res['HOTA'][idx_05]),
                'DetA': float(hota_res['DetA'][idx_05]),
                'AssA': float(hota_res['AssA'][idx_05]),
                'MOTA': float(clear_res['MOTA']),  # Aggiunto MOTA se serve
                'TP': int(clear_res['CLR_TP']),
                'FN': int(clear_res['CLR_FN']),
                'FP': int(clear_res['CLR_FP']),
            }
            detailed_results.append(row)
        except KeyError:
            continue

    def sort_key(x):
        if x['Video'] == 'GLOBAL_SCORE': return 999999
        try:
            return int(x['Video'])
        except:
            return 0

    detailed_results.sort(key=sort_key)
    return detailed_results