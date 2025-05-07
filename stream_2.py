import numpy as np
import sounddevice as sd
import soxr
from scipy import signal
import matplotlib.pyplot as plt
from fir_filter import create_fir_filter
# === Audio Configuration ===00000000000000000000
SAMPLERATE = 44100
UPSAMPLE_FACTOR = 2
UPSAMPLE_RATE = SAMPLERATE * UPSAMPLE_FACTOR
BLOCKSIZE = 1024
NUM_TAPS = 401

# Query devices and auto-detect proper channels
devices = sd.query_devices()
input_device = 3  # Your USB audio input index
output_device = 0  # HiFiBerry DAC output index

# Get actual supported channels
input_channels = min(devices[input_device]['max_input_channels'], 2)
output_channels = min(devices[output_device]['max_output_channels'], 2)

print(f"Using {input_channels} input channels and {output_channels} output channels")

# === Filter Configuration ===
FILTER_TYPE = 'lowpass'
CUTOFF = 10000
WINDOW_TYPE = ('kaiser', 10)

# Create filter
fir_coeff = create_fir_filter(
    method='remez',
    cutoff=CUTOFF,
    numtaps=NUM_TAPS,
    window_type=WINDOW_TYPE,
    filter_type=FILTER_TYPE,
    samplerate=UPSAMPLE_RATE
)

# Initialize buffers
buffer_size = NUM_TAPS + (BLOCKSIZE * UPSAMPLE_FACTOR) - 1
left_buffer = np.zeros(buffer_size, dtype=np.float32)
right_buffer = np.zeros(buffer_size, dtype=np.float32)

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(f"Stream status: {status}")
    
    # Handle mono input if needed
    if indata.ndim == 1 or indata.shape[1] == 1:
        indata = np.column_stack((indata[:,0], indata[:,0]))
    
    # Process left channel
    upsampled_left = soxr.resample(indata[:, 0], SAMPLERATE, UPSAMPLE_RATE, quality='VHQ')
    left_buffer[:-len(upsampled_left)] = left_buffer[len(upsampled_left):]
    left_buffer[-len(upsampled_left):] = upsampled_left
    processed_left = signal.fftconvolve(left_buffer, fir_coeff, mode='valid')
    outdata[:, 0] = processed_left[::UPSAMPLE_FACTOR][:frames]
    
    # Process right channel if available, else duplicate left
    if output_channels > 1:
        if input_channels > 1:
            upsampled_right = soxr.resample(indata[:, 1], SAMPLERATE, UPSAMPLE_RATE, quality='VHQ')
            right_buffer[:-len(upsampled_right)] = right_buffer[len(upsampled_right):]
            right_buffer[-len(upsampled_right):] = upsampled_right
            processed_right = signal.fftconvolve(right_buffer, fir_coeff, mode='valid')
            outdata[:, 1] = processed_right[::UPSAMPLE_FACTOR][:frames]
        else:
            outdata[:, 1] = outdata[:, 0]  # Duplicate mono to right channel

if __name__ == "__main__":
    print("Starting audio processing...")
    
    try:
        with sd.Stream(
            device=(input_device, output_device),
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            channels=(input_channels, output_channels),
            dtype='float32',
            latency='low',
            callback=audio_callback,
            extra_settings={'dither_off': True}
        ):
            print("Stream active - processing audio")
            print("Press Ctrl+C to stop")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print(f"1. Verify input device {input_device} supports at least 1 channel")
        print(f"2. Verify output device {output_device} supports stereo")
        print("3. Try setting channels=(1,2) if your input is mono")
