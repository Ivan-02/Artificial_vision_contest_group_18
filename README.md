# Artificial Vision Project - Group 18
## Soccer Player Tracking and Behavior Analysis

Questo repository contiene la soluzione del Gruppo 18 per la challenge di Artificial Vision. Il sistema esegue il tracciamento dei giocatori e il conteggio nelle ROI specificate, utilizzando YOLO per la detection e BoT-SORT per il tracking, con un filtraggio specifico per il campo da gioco.

### 📋 Requisiti
Il progetto è stato sviluppato e testato su Python 3.8+.

**Installazione delle dipendenze:**
Eseguire il seguente comando per installare tutte le librerie necessarie:
```bash
pip install -r requirements.txt
```

(Nota: Il file requirements.txt deve includere: `ultralytics , opencv-python, numpy, pandas, pyyaml, tqdm, lapx, scipy`)

---

### 📂 Struttura Cartelle Richiesta
Il sistema si aspetta la seguente struttura per i dati di input, conforme alla traccia del progetto:
```
project_root/
├── dataset/
│   └── raw_data/
│       └── test/             <-- Inserire qui le cartelle dei video di gara (es. "016", etc.)
│           ├── XXX/          <-- Id del video
│           │   ├── img1/     <-- Sequenza immagini .jpg
│           │   ├── gt/       <-- Cartella che contiene le ground thruth per track e behavior ("gt.txt", "behavior_xxx_gt.txt")
│           │   ├── gameinfo.ini
│           │   ├── roi.json  <-- File roi
│           │   └── ...
│           └── ...
├── configs/
│   ├── config.yaml           <-- Configurazione principale
│   └── tracking.yaml         <-- Configurazione tracker e filtri
│   
├── weights/
│   └── yolo11x.pt            <-- Pesi del modello (assicurarsi siano presenti)
├── src/                      <-- Codice sorgente
└── main.py                   <-- Script principale
```
---
### 🚀 Esecuzione (Setup di Gara)
Per avviare l'elaborazione completa (Tracking + Behavior Analysis) su tutti i video presenti nella cartella di test, eseguire il comando:

```
python main.py --mode contest --config configs/config.yaml
```
Descrizione del processo:

1. Inizializzazione: Lo script carica le configurazioni da `configs/config.yaml.`
2. Tracking: Esegue il tracciamento dei giocatori utilizzando YOLO e BoT-SORT, applicando un FieldFilter per rimuovere rilevamenti fuori dal campo di gioco.
3. Behavior Analysis: Analizza le posizioni dei giocatori rispetto alle ROI definite per conteggiare le presenze.
4. Output: Genera i file di testo richiesti per la sottomissione.

---
### 🎮 Simulatore e Visualizzazione Interattiva
Il progetto include uno strumento dedicato (`simulator.py`) per visualizzare graficamente i risultati generati dal Contest Runner e calcolare le metriche in tempo reale. Questo tool mostra a video il confronto tra Ground Truth (sinistra) e Predizioni (destra), utilissimo per il debug qualitativo.

Come avviare il simulatore:
\
Poiché lo script dipende da percorsi relativi, è necessario spostarsi nella cartella del simulatore prima di eseguirlo:

```bash
cd src/simulator
python simulator.py
```
Cosa fa lo script:

1. Legge la configurazione da `../../configs/config.yaml`.

2. Carica i risultati generati (file `.txt`) dalla cartella di output definita nel config (`output_contest`).

3. Apre una finestra interattiva che mostra il video con:

   * Bounding Box dei giocatori.

   * Conteggi nelle ROI (Behavior Analysis).

   * Confronto GT vs Predizioni.

4. Al termine della visualizzazione, calcola e stampa a terminale le metriche HOTA@0.5 e nMAE.

**Comandi Interattivi**:

* `Spazio`: Pausa/Riprendi video.

* `n`: Passa al video successivo.

* `q` o `Esc`: Chiudi e termina.

---
### 🛠️ Modalità Alternative di Esecuzione
Oltre alla modalità contest, è possibile eseguire i moduli singolarmente per debug o test specifici.

1. Tracking Singolo
Esegue solo la pipeline di tracking (Detection + Tracking + Field Filtering) e genera i file `tracking_X_XX.txt`. Include la visualizzazione video se `display: True` in `tracking.yaml`.

```bash
python main.py --mode track --config configs/tracking.yaml
```
2. Behavior Analysis Singolo
Esegue solo l'analisi del comportamento (conteggio nelle ROI) assumendo di avere accesso ai frame e al tracker. Genera i file `behavior_X_XX.txt`.

```bash
python main.py --mode roi --config configs/tracking.yaml
```

---
### 📊 Valutazione e Calcolo Score Finale (PTBS)
Per calcolare il punteggio finale basato sui risultati ottenuti dal run del contest, seguire questi passaggi:

**Passo 1**: Configurazione Percorso Valutazione\
Aprire il file `configs/config.yaml` e assicurarsi che la voce `output_subdirs` punti alla cartella dei risultati appena generati. Esempio (se il `test_name` in `tracking.yaml` era `track_prova_contest_runner`):

```
YAML

# In configs/config.yaml
paths:
  ...
  output_subdirs: ["track_prova_contest_runner/results"]
```
**Passo 2**: Calcolo Metrica Tracking (HOTA@0.5)\
Eseguire il comando di valutazione per il tracking:

```bash
python main.py --mode eval_hota
```
Il sistema calcolerà l'HOTA score confrontando le predizioni con la Ground Truth (se presente). Il risultato sarà stampato a video come `GLOBAL HOTA SCORE` e salvato nel file `execution_report.json` nella cartella dei risultati.

**Passo 3**: Calcolo Metrica Behavior (nMAE)\
Eseguire il comando di valutazione per il behavior:

```bash
python main.py --mode eval_roi
```

Il sistema calcolerà il Mean Absolute Error normalizzato (nMAE). Il risultato sarà stampato a video e salvato nel report.

**Passo 4**: Calcolo Punteggio Finale (PTBS)\
Il punteggio finale PTBS (Player Tracking and Behavior Score) si ottiene sommando i due valori calcolati sopra:

$$ PTBS = HOTA_{0.5} + nMAE $$

Dove:

* $ HOTA_{0.5} $: Valore ottenuto dal passo 2 (range 0-100 o 0-1, assicurarsi di usare la scala 0-1 per la somma se nMAE è in 0-1).

* nMAE: Valore ottenuto dal passo 3 (range 0-1).



---

### 📤 Output
I risultati saranno salvati automaticamente nella cartella specificata nel file `config.yaml` (default: `./output/submissions/test_name/results`).
Troverete i file formattati secondo le specifiche della traccia:

* `tracking_K_XX.txt` (es. tracking_1_18.txt): Contiene `frame_id`, `object_id` e `bounding box`.
* `behavior_K_XX.txt` (es. behavior_1_18.txt): Contiene `frame_id`, `region_id` e `conteggio giocatori`.
(Dove `18` è l'ID del nostro Team)

---

### ⚙️ Configurazione Avanzata
I parametri principali dell'algoritmo possono essere modificati nei seguenti file:

* `configs/config.yaml`: Percorsi del dataset, pesi del modello, nome del team.
* `configs/tracking.yaml`: Soglie di confidenza, IoU, parametri del Tracker (BoT-SORT) e parametri del Field Detector (filtro colore HSV e morfologico).

**Parametri Field Detector (`configs/tracking.yaml`):**
Il filtro per il rilevamento del campo da gioco può essere calibrato tramite `field_det_settings`:

-   `debug`: Impostare a `True` per visualizzare il contorno del campo rilevato.
-   `debug_mosaic`:
    -   `True`: Mostra una vista diagnostica 2x2 (Originale, HSV, Morfologica, Finale) utile per il tuning.
    -   `False`: Mostra solo il contorno verde del campo sull'immagine originale.

---
### 👥 Autori - Gruppo 18
- Simone Faraulo
- Ivan Luigi Cipriano