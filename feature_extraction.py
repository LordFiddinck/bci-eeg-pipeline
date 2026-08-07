# -*- coding: utf-8 -*- 
import math 
from typing import List, Sequence 

import numpy as np 
import pywt 
from scipy.fftpack import irfft, rfft, rfftfreq 


def fourier(low_hz: float, high_hz: float, signal: np.ndarray, fs: int) -> np.ndarray: 
    """ 
    Band-pass filter using an FFT mask 
    """ 
    freqs = rfftfreq(signal.size, d=1 / fs) 
    spectrum = rfft(signal) 

    spectrum[np.abs(freqs) > high_hz] = 0 
    spectrum[np.abs(freqs) < low_hz] = 0 

    if low_hz < 50 < high_hz: 
        spectrum[np.logical_and(np.abs(freqs) > 49, np.abs(freqs) < 51)] = 0 

    return irfft(spectrum) 


def four( 
    signal: np.ndarray, 
    fs: int, 
    band_edges: Sequence[float] = (0.5, 4, 8, 13, 20, 30, 100), 
) -> List[np.ndarray]: 
    """ 
    Split `signal` into EEG frequency bands 
    """ 
    return [ 
        fourier(band_edges[i], band_edges[i + 1], signal, fs) 
        for i in range(len(band_edges) - 1) 
    ] 


def waves(signal: np.ndarray, n_samples: int = 1200, wavelet: str = "db7") -> np.ndarray: 
    """ 
    Denoise signal via wavelet decomposition + soft thresholding 
    """ 
    coeffs = pywt.wavedec(signal, wavelet) 
    all_coeffs = np.abs(np.concatenate(coeffs)) 
    median = np.median(all_coeffs) 
    threshold = median / 0.6745 * math.sqrt(np.log(n_samples)) 

    thresholded = [pywt.threshold(c, threshold, mode="soft") for c in coeffs] 
    return pywt.waverec(thresholded, wavelet) 
