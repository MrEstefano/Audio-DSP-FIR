import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal

def plot_filter_response(coefficients, fs=44100, filter_type='lowpass', cutoff=None):
    try:
        w, h = signal.freqz(coefficients, worN=8192, fs=fs)
        
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 6))
        title = f"{filter_type} FIR Filter | Cutoff: {cutoff}Hz | Taps: {len(coefficients)}"
        fig.suptitle(title)
        
        # Magnitude plot (log scale)
        ax_mag.semilogx(w, 20*np.log10(np.abs(h) + 1e-10))
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.set_ylim(-80, 1)
        ax_mag.grid(True, which='both')
        
        # Phase plot (degrees)
        ax_phase.plot(w, np.unwrap(np.angle(h)) * 180/np.pi)
        ax_phase.set_xlabel('Frequency (Hz)')
        ax_phase.set_ylabel('Phase (degrees)')
        ax_phase.set_xlim(0, fs//2)
        ax_phase.grid(True)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
    except Exception as e:
        print(f"Plotting error: {e}")
