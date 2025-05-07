import numpy as np
import sounddevice as sd
import soxr
from scipy import signal
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from fir_filter import create_fir_filter
from plot_filter import plot_filter_response

# === Audio Configuration ===
SAMPLERATE = 44100
UPSAMPLE_FACTOR = 3  # Changed to 2x
UPSAMPLE_RATE = SAMPLERATE * UPSAMPLE_FACTOR  # Now 88.2kHz
CHANNELS = 1
BLOCKSIZE = 1024
NUM_TAPS = 301  # Must be odd

# === Filter Configuration ===
FILTER_TYPE = 'lowpass'
CUTOFF = 12000  # Below Nyquist (44.1kHz)
WINDOW_TYPE = ('kaiser', 10)

# Create filter
fir_coeff = create_fir_filter(
    method='remez',
    cutoff=CUTOFF,
    numtaps=NUM_TAPS,
    window_type=WINDOW_TYPE,
    filter_type=FILTER_TYPE,
    samplerate=UPSAMPLE_RATE,
    min_phase=False,
    #weight=[1, 20]  # Strong stopband emphasis
)

# Plot filter response
plot_filter_response(fir_coeff, fs=UPSAMPLE_RATE, filter_type=FILTER_TYPE)

# === Buffer Setup ===
input_buffer_size = NUM_TAPS + (BLOCKSIZE * UPSAMPLE_FACTOR) - 1
input_buffer = np.zeros(input_buffer_size, dtype=np.float32)
print(f"Buffer size: {input_buffer_size}")  # Should be 801 + 2048 - 1 = 2848

def apply_dither(audio, bit_depth=24):
    """Apply TPDF dithering"""
    dither = (np.random.random(len(audio)) - 0.5) * (2 / (2**bit_depth))
    return audio + dither

def audio_callback(indata, outdata, frames, time, status):
    global input_buffer
    
    if status:
        print(f"Stream status: {status}")
    
    # 1. Upsample (1024 ? 2048 samples)
    upsampled = soxr.resample(
        indata[:, 0],
        SAMPLERATE,
        UPSAMPLE_RATE,
        quality='VHQ'
    )
    
    # 2. Validate sizes
    if len(upsampled) != BLOCKSIZE * UPSAMPLE_FACTOR:
        print(f"Warning: Expected {BLOCKSIZE*UPSAMPLE_FACTOR} samples, got {len(upsampled)}")
    
    # 3. Update buffer (efficient roll)
    input_buffer[:-len(upsampled)] = input_buffer[len(upsampled):]
    input_buffer[-len(upsampled):] = upsampled
    
    # 4. Process with FFT convolution
    processed = signal.fftconvolve(input_buffer, fir_coeff, mode='valid')
    
    # 5. Downsample (2048 ? 1024 samples)
    downsampled = processed[::UPSAMPLE_FACTOR][:frames]
    
    # 6. Output
    outdata[:, 0] = apply_dither(downsampled)

if __name__ == "__main__":
    print(f"Starting DSP processing with {UPSAMPLE_FACTOR}x upsampling...")
    print(f"Input buffer size: {len(input_buffer)}")
    print(f"Upsampled block size: {BLOCKSIZE * UPSAMPLE_FACTOR}")
    
    try:
        with sd.Stream(
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            dtype='float32',
            latency='high',
            callback=audio_callback,
            device=(1, 0)
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")
