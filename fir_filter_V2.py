import numpy as np
from scipy import signal

def create_fir_filter(
    method='remez',  # Default to remez now
    cutoff=12000,
    numtaps=511,
    window_type=('kaiser', 10.0),
    filter_type='lowpass',
    samplerate=44100,
    min_phase=False,
    weight=None
):
    nyq = 0.5 * samplerate
    
    if isinstance(cutoff, list):  # Bandpass case
        bands = [0, cutoff[0]*0.9, cutoff[0], cutoff[1], cutoff[1]*1.1, nyq]
        desired = [0, 0, 1, 1, 0] if filter_type == 'bandpass' else [1, 1, 0, 0, 1]
    else:  # Lowpass/Highpass
        transition = min(2000, cutoff*0.2)  # Dynamic transition width
        if filter_type == 'lowpass':
            bands = [0, cutoff, cutoff + transition, nyq]
            desired = [1, 1, 0, 0]
            weight = weight or [1, 10]  # Emphasize stopband
        else:  # highpass
            bands = [0, cutoff - transition, cutoff, nyq]
            desired = [0, 0, 1, 1]
    
    if method == 'remez':
        coeff = signal.remez(
            numtaps,
            bands,
            desired,
            fs=samplerate,
            weight=weight
        )
    else:
        coeff = signal.firwin(
            numtaps,
            cutoff,
            window=window_type,
            pass_zero=filter_type,
            fs=samplerate
        )
    
    return coeff.astype(np.float32)
