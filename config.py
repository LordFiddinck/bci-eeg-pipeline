import os 
from pathlib import Path 

ROOT = Path(__file__).resolve().parent # the folder this config.py lives in, one level up 
DATA_DIR = Path(os.environ.get("BCI_DATA_DIR", ROOT / "data")) 
MODEL_DIR = Path(os.environ.get("BCI_MODEL_DIR", ROOT / "models")) 
LOG_DIR = Path(os.environ.get("BCI_LOG_DIR", ROOT / "logs")) 

for d in (DATA_DIR, MODEL_DIR, LOG_DIR): 
    d.mkdir(parents=True, exist_ok=True) 

SUBJECT_ID = os.environ.get("BCI_SUBJECT_ID", "Subject_45") # default for testing 

EEG_CSV_PATH = DATA_DIR / f"{SUBJECT_ID}.csv" 
KEY_LOG_PATH = DATA_DIR / f"{SUBJECT_ID}.txt" 
MAZE_LOG_PATH = DATA_DIR / f"{SUBJECT_ID}_Maze_Log.txt" 
PROCESSED_PICKLE_PATH = DATA_DIR / f"{SUBJECT_ID}_Processed.pickle" 

TRAIN_SET_GLOB = "EEG_Dataset*.pickle" 

# Signal parameters 
FS = 256  # sampling rate in Hz
ELECTRODE_LABELS = [ 
    "Cz", "Fz", "Fp1", "F7", "F3", "FC1", "C3", "FC5", "FT9", 
    "T7", "T8", "FT10", "FC6", "C4", "FC2", "F4", "F8", "Fp2", 
] 

# Electrode coordinates on a 22x43 interpolation 
COORDINATE_MAPPING = { 
    "Cz": (21, 21), "Fz": (13, 21), "Fp1": (5, 16), "F7": (11, 8), 
    "F3": (12, 14), "FC1": (17, 17), "C3": (21, 13), "FC5": (16, 9), 
    "FT9": (15, 1), "T7": (21, 4), "T8": (21, 38), "FT10": (15, 41), 
    "FC6": (16, 33), "C4": (21, 29), "FC2": (17, 25), "F4": (12, 28), 
    "F8": (11, 34), "Fp2": (5, 26), 
} 

KEY_LABELS = ["Key.up", "Key.down", "Key.left", "Key.right"] 

# Time window (ms) taken around each key press for classification 
TAKEN_RT_MS = 500 
MATRIX_TS = round(TAKEN_RT_MS * FS / 1000) 

INFERENCE_API_URL = os.environ.get("BCI_API_URL", "http://localhost:4224/bci") 
INFERENCE_API_KEY = os.environ.get("BCI_API_KEY", "") 

ENTRIES_PER_SECOND = 256 # samples/sec set by headset 
BUFFER_SECONDS = 2.0 
ELECTRODE_STREAM_IDS = [0, 6, 27]  # Cz, C3, C4 indices within the raw LSL sample (testing) 
