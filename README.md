# EEG-based BCI Pipeline 

Code accompanying the dissertation for the MSc MedTech Innovation and 
Entrepreneurship final project, on an sffordable BCI device to replace 
gaming peripherals with thought decoding (using non-invasive EEG)


The goal was to find how few EEG electrodes a non-invasive, non-gel headset 
needs to reliably read a user's intent to move. This repository is the 
pipeline usedto answer that question it turns raw scalp EEG into a predicted 
command (in this case, LEFT ARM / RIGHT ARM / NO ARM) and includes the
offline preprocessing and training code that produced the underlying model 

## Key result 

Across 45 participants wearing an EMOTIV Flex 2 Saline Wireless Cap, deep 
neural networks trained on this pipeline distinguished: 

- Distinguish between directions: 83.3% 
- Distinguish movement from steady state: 80.9% 
- Distinguish between movements performed: 78.5% 
- Distinguish movement from imagination: 68.75% 

With insufficient separation for: 

- Distinguish between movements imagined: 57.1% 
- Distinguish wrong from correct input: 56.25% 
- Distinguish between different intentions: 53.8% 

The best-performing tasks needed as few as three electrodes (C3, C4, Cz over 
the motor cortex), with a second three-electrode set (Fz, F3, F4) used for 
the imagination/error-related-potential tasks. The resulting model is light 
enough (~300 MB, ~10 ms inference) to run on consumer hardware in real time 

Full methodology, experiment design, and results discussion are in the
dissertation (not included in this repo) 

## Files 

``` 
config.py               Shared paths, constants, and environment-variable 
                         overrides (subject ID, data/model/log dirs, electrode 
                         layout, API endpoint) 
preprocessing.py        Loads a raw EEG export + keystroke/event log for one 
                         session, cleans channels, time-aligns EEG samples to 
                         key events, and segments the session into its tasks 
feature_extraction.py   Frequency-band filtering and wavelet denoising, 
                         shared by preprocessing and training 
matrix_support.py       Interpolates sparse electrode readings into a dense 
                         2D grid per timestep for a 2D CNN to treat an EEG 
                         sample like an image 
train_model.py          Trains one binary "SingularNet" CNN per command 
                         class, in a competitive scheme where each step 
                         reinforces the correct class and suppresses a 
                         randomly-chosen wrong one 
eeg_stream.py           Real-time client: pulls EEG from an LSL stream, 
                         buffers a rolling window, sends it to an inference 
                         API containing a trained model, and prints the
                         predicted command 
requirements.txt        Python dependencies 
``` 

## Pipeline 

``` 
eeg_stream.py           live EEG (LSL) → buffered window → inference API → predicted command 
preprocessing.py        raw EEG + keystroke log → time-aligned, task-segmented session data 
feature_extraction.py   segmented data → frequency-band filtering, wavelet denoising 
matrix_support.py       electrode readings → dense interpolated 2D grid (CNN input) 
train_model.py          featurized + gridded data → 4 per-class CNNs ("SingularNet" ensemble) 
``` 

## Requirements 

- Python 3.10+ 
- Install dependencies with: 
```bash 
  pip install -r requirements.txt 
``` 
- `eeg_stream.py` additionally expects: 
  - An LSL-compatible EEG headset/relay broadcasting a stream of `type="EEG"` 
    that `pylsl` can resolve on the local network 
  - A running inference API (not included in this repo) reachable at 
    `BCI_API_URL`, returning JSON of the form `{"output": [...]}` 
    (where trained models would be placed for live sessions) 
- **Data is not included in this repository.** The raw EEG exports and key 
  logs were collected under the study protocol and are not public. 
  `preprocessing.py` and `train_model.py` will only run once 
  you've pointed the config at collected data of the same shape (see below) 

## Running the code 

All paths and IDs are controlled via environment variables read in `config.py`: 

| Variable         | Default              | Purpose                                   | 
|-------------------|----------------------|--------------------------------------------| 
| `BCI_DATA_DIR`    | `./data`             | Folder holding raw EEG/key-log exports and processed pickles | 
| `BCI_MODEL_DIR`   | `./models`           | Where trained model checkpoints are written | 
| `BCI_LOG_DIR`     | `./logs`             | Reserved for run logs | 
| `BCI_SUBJECT_ID`  | `Subject_45`         | Selects `{SUBJECT_ID}.csv` / `{SUBJECT_ID}.txt` / `{SUBJECT_ID}_Maze_Log.txt` in `BCI_DATA_DIR` | 
| `BCI_API_URL`     | `http://localhost:4224/bci` | Inference API endpoint used by `eeg_stream.py` | 
| `BCI_API_KEY`     | *(empty)*            | API key sent with each inference request  | 

**1. Preprocess one session** 

Place `{SUBJECT_ID}.csv` (raw EEG export) and `{SUBJECT_ID}.txt` (key-press 
log) in `BCI_DATA_DIR`, then run: 

```bash 
export BCI_SUBJECT_ID=Subject_01 
export BCI_DATA_DIR=/path/to/data 
python preprocessing.py 
``` 

This cleans and time-aligns the session and writes 
`{SUBJECT_ID}_Processed.pickle` into `BCI_DATA_DIR`. Note that 
`align_keys_to_eeg` parses the key log using the exact field 
offsets and key bindings from data collection (see Limitations) 

**2. Assemble a training set** 

`train_model.py` expects one or more `EEG_Dataset*.pickle` files in 
`BCI_DATA_DIR`, each containing a list of `(raw_dataframe, feature_dataframe, 
key_label)` tuples across sessions/subjects. Assembling these from the 
per-session output of step 1 is not automated here and will depend on how 
sessions are grouped 

**3. Train** 

```bash 
python train_model.py 
``` 

Runs for `EPOCHS` (default 300), validating every `VAL_EVERY_N_EPOCHS` 
epochs, and after every epoch saves one checkpoint per key class to 
`BCI_MODEL_DIR/model_<label>.pt`, plus `BCI_MODEL_DIR/history.pickle` with 
loss/accuracy curves. Device placement isn't pinned in the code (add 
`.to(device)` for GPU training) 

**4. Real-time inference** 

```bash 
export BCI_API_URL=http://your-inference-host:4224/bci 
export BCI_API_KEY=your-key 
python eeg_stream.py 
``` 

Resolves a live LSL EEG stream, buffers a rolling `BUFFER_SECONDS` window of 
the 3 electrodes in `ELECTRODE_STREAM_IDS`, and POSTs each window to 
`BCI_API_URL` for a prediction, printed to stdout along with round-trip 
latency. This script assumes an inference server that loads the checkpoints 
from step 3 and serves predictions 

## Limitations

This code reflects a dissertation-stage exploratory prototype 

- **Sample size and demographics.** Results come from 45 healthy, 
  predominantly young participants, not the clinical/healthcare population 
  the long-term device is aimed at 
- **Three of the seven hypotheses were not confirmed at the pre-registered confidence threshold.** 
  Distinguishing wrong from correct input, imagined movement type, and intention 
  type. The accuracy numbers above are the strongest results, without viable 
  augmentation or cross-section validation 
- **`preprocessing.py`'s task-boundary detection is tightly coupled** to the 
  exact log format used during data collection (character-offset-based parsing 
  of specific fields). It will need adaptation for different logging setups 

## Original Dissertation: 

> Caiado, F., Ukolov, A., Liu, C., Cakti, R. A., & Djari, D. (2024). *The 
> Power of the Mind: Research, Development and Business Plan for a Device to 
> Control Technology through Thought*. King's College London 
