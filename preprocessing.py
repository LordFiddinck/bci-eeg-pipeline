import datetime 
import pickle 
import time 

import numpy as np 
import pandas as pd 

from config import EEG_CSV_PATH, KEY_LOG_PATH, MAZE_LOG_PATH, PROCESSED_PICKLE_PATH 


def load_raw_eeg(csv_path=EEG_CSV_PATH) -> pd.DataFrame: 
    if not csv_path.exists(): 
        raise FileNotFoundError( 
            f"No EEG export found at {csv_path}. Set BCI_SUBJECT_ID / " 
            f"BCI_DATA_DIR in your environment, or place the file there." 
        ) 
    return pd.read_csv(csv_path, skiprows=1) 


def clean_eeg_channels(eeg_data: pd.DataFrame) -> pd.DataFrame: 
    """ 
    Drop channels that are entirely NaN or entirely zero for their first 
    100 samples (dead/disconnected electrodes), then re-reference every 
    remaining channel to the mean of all channels at each timestep 
    """ 
    probe = eeg_data.iloc[:100] 
    all_nan = probe.isna().all() 
    all_zero = (probe == 0).all() 
    eeg_data = eeg_data.loc[:, ~(all_nan | all_zero)] 

    # Keep the timestamp column plus short-named "EEG*" channels only 
    # (the original also dropped e.g. "EEG_Quality" style long names). 
    keep_cols = [ 
        col for col in eeg_data.columns 
        if col == eeg_data.columns[0] or (col.startswith("EEG") and len(col) < 9) 
    ] 
    dataset = eeg_data[keep_cols].copy() 

    channel_cols = [c for c in dataset.columns if c.startswith("EEG")] 
    mean_signal = dataset[channel_cols].mean(axis=1) 
    for col in channel_cols: 
        dataset[col] = dataset[col] - mean_signal 

    dataset["Key Pressed"] = "None" 
    return dataset 


def _parse_log_timestamp(line: str) -> float: 
    fmt = "%Y-%m-%d %H:%M:%S" 
    date = datetime.datetime.strptime(line[0:19], fmt) 
    milliseconds = int(line[20:23]) 
    date = date.replace(microsecond=milliseconds * 1000) 
    return time.mktime(date.timetuple()) + milliseconds / 1000 


def align_keys_to_eeg(dataset: pd.DataFrame, log_path=KEY_LOG_PATH): 
    """ 
    Walk the key-press/release log line by line, find the EEG sample whose 
    timestamp is closest at-or-after each event, and stamp that row's 
    "Key Pressed" column. Also tracks the first occurrence of specific 
    task-marker keys so the session can later be sliced into Task 0-4 
    """ 
    tasks = { 
        "Task 0": 0, 
        "Task 1": {"Move 1": 0, "Move 2": 0, "Move 3": 0, "Move 4": 0}, 
        "Task 2": {"Move 1": 0, "Move 2": 0, "Move 3": 0, "Move 4": 0, "Move 5": 0}, 
        "Task 3 and 4": 0, 
    } 

    cur_index = 0 
    previous_char = "" 

    with open(log_path, "r") as log: 
        for next_line in log: 
            if next_line[0:4] != "2024": 
                continue 

            unix_time = _parse_log_timestamp(next_line) 

            press_marker = next_line.find("pressed") 
            if press_marker > 0: 
                while cur_index < dataset.shape[0] - 1 and float(dataset["Timestamp"].loc[cur_index]) < unix_time: 
                    cur_index += 1 

                key_press = next_line[25:press_marker - 1].replace("'", "").replace(" ", "") 
                if "ABDCEFGHIJKLMNOPQRSTUVWXYZ".find(key_press) >= 0: 
                    key_press = key_press.casefold() 

                if key_press == "k" and tasks["Task 1"]["Move 1"] == 0 and key_press != previous_char: 
                    tasks["Task 1"]["Move 1"] = cur_index 
                if key_press == "f" and tasks["Task 1"]["Move 2"] == 0 and tasks["Task 1"]["Move 1"] != 0 and key_press != previous_char: 
                    tasks["Task 1"]["Move 2"] = cur_index 
                if key_press == "a" and key_press != previous_char and tasks["Task 1"]["Move 1"] != 0: 
                    if tasks["Task 1"]["Move 3"] == 0: 
                        tasks["Task 1"]["Move 3"] = cur_index 
                    elif tasks["Task 2"]["Move 2"] == 0: 
                        tasks["Task 2"]["Move 2"] = cur_index 
                if key_press == "e" and tasks["Task 1"]["Move 4"] == 0 and tasks["Task 1"]["Move 2"] != 0 and key_press != previous_char: 
                    tasks["Task 1"]["Move 4"] = cur_index 
                if key_press == "u" and tasks["Task 2"]["Move 1"] == 0 and tasks["Task 1"]["Move 2"] != 0 and key_press != previous_char: 
                    tasks["Task 2"]["Move 1"] = cur_index 
                if key_press == "t" and tasks["Task 2"]["Move 3"] == 0 and tasks["Task 1"]["Move 2"] != 0 and key_press != previous_char: 
                    tasks["Task 2"]["Move 3"] = cur_index 
                if key_press == "q" and tasks["Task 2"]["Move 4"] == 0 and tasks["Task 1"]["Move 2"] != 0 and key_press != previous_char: 
                    tasks["Task 2"]["Move 4"] = cur_index 
                if key_press == "d" and tasks["Task 2"]["Move 5"] == 0 and tasks["Task 1"]["Move 2"] != 0 and key_press != previous_char: 
                    tasks["Task 2"]["Move 5"] = cur_index 
                if key_press == "g" and tasks["Task 3 and 4"] == 0 and tasks["Task 1"]["Move 2"] != 0 and key_press != previous_char: 
                    tasks["Task 3 and 4"] = cur_index 

                if dataset["Key Pressed"].loc[cur_index].startswith("Released"): 
                    cur_index += 1 
                if key_press != "Key.caps_lock": 
                    dataset.at[cur_index, "Key Pressed"] = key_press 

                previous_char = key_press 

            release_marker = next_line.find("release") 
            if release_marker > 0: 
                while cur_index < dataset.shape[0] - 1 and float(dataset["Timestamp"].loc[cur_index]) < unix_time: 
                    cur_index += 1 
                key_release = "Released " + next_line[25:release_marker - 1].replace("'", "").replace(" ", "") 
                if key_release.endswith(("Key.left", "Key.right", "Key.up", "Key.down")): 
                    if dataset["Key Pressed"].loc[cur_index] != "None": 
                        cur_index += 1 
                    dataset.at[cur_index, "Key Pressed"] = key_release 

    _assert_boundaries_found(tasks) 
    return dataset, tasks 


def _assert_boundaries_found(tasks: dict): 
    """ 
    Warns if a task boundary was never matched in the log, instead 
    of silently slicing the dataset at index 0 later on 
    """ 
    unresolved = [] 
    for move, idx in tasks["Task 1"].items(): 
        if idx == 0: 
            unresolved.append(f"Task 1 / {move}") 
    for move, idx in tasks["Task 2"].items(): 
        if idx == 0: 
            unresolved.append(f"Task 2 / {move}") 
    if tasks["Task 3 and 4"] == 0: 
        unresolved.append("Task 3 and 4") 
    if unresolved: 
        raise ValueError( 
            "The following task boundaries were never found while parsing " 
            f"the key log, which usually means the log format or key " 
            f"bindings changed: {unresolved}" 
        ) 


def compute_reaction_time(task0: pd.DataFrame) -> float: 
    """ 
    Reaction time = time between "GO" cue and space/click response, 
    filtered to plausible (2.5-4s) responses, averaged over the last 3 
    """ 
    previous_time, set_of_times = 0, [] 

    for index in range(task0.shape[0]): 
        key = task0["Key Pressed"].loc[index] 
        timestamp = task0["Timestamp"].loc[index] 
        if key != "None": 
            if key in ("Key.space", "Button.left"): 
                set_of_times.append((key, timestamp - previous_time)) 
            previous_time = timestamp 

    recent_spaces = [t for k, t in reversed(set_of_times) if k == "Key.space"][:3] 
    plausible = [t for t in recent_spaces if 2.5 < t < 4] 

    if not plausible: 
        raise ValueError("Could not compute a plausible reaction time from Task 0.") 
    return float(np.mean(plausible)) 


def save_processed(data_dict: dict, path=PROCESSED_PICKLE_PATH): 
    path.parent.mkdir(parents=True, exist_ok=True) 
    with open(path, "wb") as fh: 
        pickle.dump(data_dict, fh) 
    print(f"Saved processed session to {path}") 


def run_pipeline(): 
    eeg = load_raw_eeg() 
    eeg = clean_eeg_channels(eeg) 
    eeg, tasks = align_keys_to_eeg(eeg) 

    task0_end = tasks["Task 1"]["Move 1"] 
    task0 = eeg[:task0_end] 
    reaction_time_from_key = compute_reaction_time(task0) 
    reaction_time_from_go = reaction_time_from_key - 3 

    print(f"Reaction time (from key): {reaction_time_from_key:.3f}s") 
    print(f"Reaction time (from GO cue): {reaction_time_from_go:.3f}s") 

    data_dict = { 
        "Original": { 
            "Task 0": {"RT From GO": reaction_time_from_go, "RT From Key": reaction_time_from_key}, 
            "Task boundaries": tasks, 
        }, 
        "Pre-Processed": {}, 
        "Features": {}, 
    } 
    save_processed(data_dict) 
    return data_dict 


if __name__ == "__main__": 
    run_pipeline() 
