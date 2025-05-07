import numpy as np
from scipy import signal

def create_fir_filter(
    method='firwin',
    cutoff=10000,
    numtaps=401,
    window_type=('kaiser', 10),
    filter_type='lowpass',
    samplerate=88200,
    min_phase=False
):
    nyq = 0.5 * samplerate
    
    # Calculate conservative transition width (30% of cutoff or 2500Hz max)
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        if len(cutoff) != 2:
            raise ValueError("Bandpass/bandstop requires exactly 2 cutoff frequencies")
        transition = min(2500, 0.3 * min(cutoff))
        f1, f2 = cutoff[0], cutoff[1]
    else:
        transition = min(2500, 0.3 * cutoff)
        f1, f2 = cutoff, cutoff  # Not used for lowpass/highpass
    
    # Design parameters for all filter types
    if filter_type == 'lowpass':
        bands = [0, cutoff, cutoff+transition, nyq]
        desired = [1, 0]
        weight = [1, 10]
        firwin_pass_zero = True
    elif filter_type == 'highpass':
        bands = [0, cutoff-transition, cutoff, nyq]
        desired = [0, 1]
        weight = [10, 1]
        firwin_pass_zero = False
    elif filter_type == 'bandpass':
        if not isinstance(cutoff, (list, tuple, np.ndarray)):
            raise ValueError("Bandpass requires two cutoff frequencies")
        bands = [0, f1-transition, f1, f2, f2+transition, nyq]
        desired = [0, 0, 1, 1]
        weight = [1, 10, 1]
        firwin_pass_zero = False
    elif filter_type == 'bandstop':
        if not isinstance(cutoff, (list, tuple, np.ndarray)):
            raise ValueError("Bandstop requires two cutoff frequencies")
        bands = [0, f1-transition, f1, f2, f2+transition, nyq]
        desired = [1, 1, 0, 0]
        weight = [10, 1, 10]
        firwin_pass_zero = True
    else:
        raise ValueError(f"Invalid filter_type '{filter_type}'")

    # Filter design with fallback to firwin if remez fails
    if method == 'remez':
        try:
            return signal.remez(
                numtaps,
                bands,
                desired,
                weight=weight,
                fs=samplerate,
                maxiter=100
            ).astype(np.float32)
        except Exception as e:
            print(f"remez failed ({str(e)}), falling back to firwin")
            method = 'firwin'
    
    if method == 'firwin':
        if isinstance(cutoff, (list, tuple, np.ndarray)):
            # For bandpass/bandstop, use average cutoff for firwin
            cutoff = (f1 + f2) / 2
        return signal.firwin(
            numtaps,
            cutoff,
            width=transition,
            window=window_type,
            pass_zero=firwin_pass_zero,
            fs=samplerate
        ).astype(np.float32)
    
    raise ValueError("method must be 'remez' or 'firwin'")
