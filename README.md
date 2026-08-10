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
eeg_stream.py            Real-time client: pulls EEG from an LSL stream, 
                         buffers a rolling window, sends it to an inference 
                         API, and prints the predicted command 
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
