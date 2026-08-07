import asyncio 
import time 

import numpy as np 
import requests 
from pylsl import StreamInlet, resolve_stream 

from config import ( 
    BUFFER_SECONDS, 
    ELECTRODE_STREAM_IDS, 
    ENTRIES_PER_SECOND, 
    INFERENCE_API_KEY, 
    INFERENCE_API_URL, 
) 

BUFFER_SIZE_SAMPLES = int(ENTRIES_PER_SECOND * BUFFER_SECONDS) 


async def send_to_network(window: list) -> list | None: 
    """ 
    Post a buffered window of samples to the inference API 
    """ 
    payload = {"data": str(window), "apikey": INFERENCE_API_KEY} 

    try: 
        # requests is synchronous; run it off the event loop so we don't 
        # block other coroutines while waiting on the network. 
        response = await asyncio.to_thread(requests.post, INFERENCE_API_URL, data=payload) 
    except requests.RequestException as exc: 
        print(f"Network error contacting inference API: {exc}") 
        return None 

    if response.status_code == 200: 
        return response.json().get("output") 

    print(f"Error {response.status_code}: {response.content}") 
    return None 


def interpret_prediction(result) -> str: 
    """ 
    Map the raw model output vector to a label 
    """ 
    result_arr = np.asarray(result) 
    best_id = int(np.argmax(result_arr)) 
    best_value = round(float(result_arr.flat[best_id])) 

    if best_value < 1: 
        return "NO ARM" 
    return {0: "LEFT ARM", 1: "RIGHT ARM"}.get(best_id, f"UNKNOWN ({best_id})") 


async def stream_eeg(inlet: StreamInlet, buffer: asyncio.Queue, stop_event: asyncio.Event): 
    """ 
    Continuously pull samples from the LSL inlet and push the electrodes 
    of interest onto the buffer until stop_event 
    """ 
    while not stop_event.is_set(): 
        sample, timestamp = inlet.pull_sample(timeout=1.0) 
        if timestamp is None: 
            continue  # no sample ready yet; keep polling 

        sample = np.asarray(sample[3:-2])  # trim non-EEG header/trailer channels 
        line = [sample[electrode] for electrode in ELECTRODE_STREAM_IDS] 
        await buffer.put(line) 


async def check_network_response(buffer: asyncio.Queue, stop_event: asyncio.Event): 
    """ 
    Whenever the buffer holds a full window, pull the oldest entries, 
    send the window to the API, and report the prediction 
    """ 
    while not stop_event.is_set(): 
        if buffer.qsize() < BUFFER_SIZE_SAMPLES: 
            await asyncio.sleep(0.05) 
            continue 

        # Drop any stale surplus beyond one window's worth. 
        while buffer.qsize() > BUFFER_SIZE_SAMPLES: 
            buffer.get_nowait() 

        window = [buffer.get_nowait() for _ in range(BUFFER_SIZE_SAMPLES)] 

        time_began = time.time() 
        result = await send_to_network(window) 
        time_end = time.time() 

        if result is None: 
            print("No result from API this cycle; skipping.") 
            continue 

        print("RESULT:", interpret_prediction(result)) 
        print("RESULT RAW:", result) 
        print("LATENCY:", round(time_end - time_began, 4), "s") 


async def main(): 
    print("Resolving EEG stream...") 
    streams = resolve_stream("type", "EEG") 
    inlet = StreamInlet(streams[0]) 

    buffer: asyncio.Queue = asyncio.Queue() 
    stop_event = asyncio.Event() 

    stream_task = asyncio.create_task(stream_eeg(inlet, buffer, stop_event)) 
    predict_task = asyncio.create_task(check_network_response(buffer, stop_event)) 

    try: 
        await asyncio.gather(stream_task, predict_task) 
    except asyncio.CancelledError: 
        pass 
    finally: 
        stop_event.set() 
        print("Stream stopped.") 


if __name__ == "__main__": 
    try: 
        asyncio.run(main()) 
    except KeyboardInterrupt: 
        print("\nInterrupted by user, shutting down.") 
