import numpy as np
import sounddevice as sd
import soxr
from scipy import signal
import matplotlib.pyplot as plt
from fir_filter import create_fir_filter

# === Audio Configuration ===
SAMPLERATE = 44100
UPSAMPLE_FACTOR = 2
UPSAMPLE_RATE = SAMPLERATE * UPSAMPLE_FACTOR
CHANNELS = 2  # Stereo
BLOCKSIZE = 1024
NUM_TAPS = 401  # Must be odd
DEVICE = 'hw:0,0'  # Explicitly use HiFiBerry DAC

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

# Plot configuration
plt.figure()
w, h = signal.freqz(fir_coeff, worN=8000, fs=UPSAMPLE_RATE)
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title('Filter Frequency Response')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.grid(True)
plt.show(block=False)

# === Audio Buffers ===
buffer_size = NUM_TAPS + (BLOCKSIZE * UPSAMPLE_FACTOR) - 1
print(f"Buffer size per channel: {buffer_size}")
left_buffer = np.zeros(buffer_size, dtype=np.float32)
right_buffer = np.zeros(buffer_size, dtype=np.float32)

def apply_dither(audio, bit_depth=24):
    dither = (np.random.random(len(audio)) - 0.5) * (2 / (2**bit_depth))
    return audio + dither

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(f"Stream status: {status}")
    
    # Process left channel
    upsampled_left = soxr.resample(indata[:, 0], SAMPLERATE, UPSAMPLE_RATE, quality='VHQ')
    left_buffer[:-len(upsampled_left)] = left_buffer[len(upsampled_left):]
    left_buffer[-len(upsampled_left):] = upsampled_left
    processed_left = signal.fftconvolve(left_buffer, fir_coeff, mode='valid')
    outdata[:, 0] = apply_dither(processed_left[::UPSAMPLE_FACTOR][:frames])
    
    # Process right channel
    upsampled_right = soxr.resample(indata[:, 1], SAMPLERATE, UPSAMPLE_RATE, quality='VHQ')
    right_buffer[:-len(upsampled_right)] = right_buffer[len(upsampled_right):]
    right_buffer[-len(upsampled_right):] = upsampled_right
    processed_right = signal.fftconvolve(right_buffer, fir_coeff, mode='valid')
    outdata[:, 1] = apply_dither(processed_right[::UPSAMPLE_FACTOR][:frames])

if __name__ == "__main__":
    print("Starting stereo DSP processing on HiFiBerry DAC...")
    
    try:
        # Configure stream with explicit device
        with sd.Stream(
            device=(DEVICE, DEVICE),  # Use same device for input/output
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            dtype='float32',
            latency='high',
            callback=audio_callback,
            extra_settings={'number_of_input_channels': 0}  # Force output-only mode
        ):
            print("Stream started successfully")
            while True:
                sd.sleep(1000)
    except Exception as e:
        print(f"Error: {e}")
        print("\nAdditional troubleshooting:")
        print("1. Run 'sudo raspi-config' and ensure audio is set to HiFiBerry")
        print("2. Check '/boot/config.txt' for correct dtoverlay=hifiberry-dac")
        print("3. Test with 'aplay -D hw:0,0 /usr/share/sounds/alsa/Front_Center.wav'")
