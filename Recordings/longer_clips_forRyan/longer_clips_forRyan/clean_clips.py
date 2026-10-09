import os
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------
#  Band-Pass Filter
# ---------------------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, fs, lowcut, highcut):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

# ---------------------------------------------------------
#  Noise Reduction (Spectral Gating)
# ---------------------------------------------------------
def reduce_noise(y, sr, noise_seconds=0.3):
    n_samples = int(noise_seconds * sr)
    noise_clip = y[:n_samples]

    n_fft = 2048
    hop = 512

    noise_stft = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop)
    noise_mag = np.abs(noise_stft)
    noise_thresh = np.mean(noise_mag, axis=1) * 1.5

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag = np.abs(stft)
    phase = np.angle(stft)

    mask = mag >= noise_thresh[:, None]
    mag_clean = mag * mask

    cleaned_stft = mag_clean * np.exp(1j * phase)
    cleaned = librosa.istft(cleaned_stft, hop_length=hop)

    return cleaned

# ---------------------------------------------------------
#  Clean Single File
# ---------------------------------------------------------
def clean_cricket_audio(input_path, output_path, lowcut, highcut):
    y, sr = librosa.load(input_path, sr=None)
    filtered = bandpass_filter(y, sr, lowcut, highcut)
    cleaned = reduce_noise(filtered, sr)
    cleaned = cleaned / np.max(np.abs(cleaned))
    sf.write(output_path, cleaned, sr)
    print("Saved:", output_path)

# ---------------------------------------------------------
#  Recursive Folder Walker (Preserves Structure)
# ---------------------------------------------------------
def clean_recursive(input_root, output_root,
                    lowcut=3000, highcut=7000):

    for root, dirs, files in os.walk(input_root):
        # Compute matching output directory
        rel_path = os.path.relpath(root, input_root)
        out_dir = os.path.join(output_root, rel_path)

        # Create output directory
        os.makedirs(out_dir, exist_ok=True)

        # Process WAV files in this folder
        for f in files:
            if f.lower().endswith(".wav"):
                input_file = os.path.join(root, f)
                output_file = os.path.join(out_dir, f)

                clean_cricket_audio(
                    input_file,
                    output_file,
                    lowcut,
                    highcut
                )

    print("\nAll folders processed successfully.\n")


# ---------------------------------------------------------
#  RUN IT
# ---------------------------------------------------------
clean_recursive(
    input_root="/data/larsonlab/ihudsonfoy/allsongs1/chosen_random_clips",
    output_root="/data/larsonlab/ihudsonfoy/allsongs1/chosen_random_clips/cleaned",
    lowcut=3500,   # adjust for species
    highcut=6500
)
