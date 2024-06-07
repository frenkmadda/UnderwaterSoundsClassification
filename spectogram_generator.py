import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import gc
from memory_profiler import profile

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    return audio, sr

def calculate_spectrogram(audio, sample_rate):
    mono_audio = librosa.to_mono(audio)
    stft = librosa.stft(mono_audio)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return spectrogram

def visualize_audio(audio, sample_rate, output_file):
    spectrogram = calculate_spectrogram(audio, sample_rate)
    fig = plt.figure(figsize=(12, 6))
    img = librosa.display.specshow(spectrogram, sr=sample_rate, x_axis="time", y_axis="log")
    plt.axis('off')
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
    plt.colorbar().remove()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

@profile
def process_audio_file(file_path, output_dir):
    audio, sample_rate = load_audio(file_path)
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.png')
    visualize_audio(audio, sample_rate, output_file)
    del audio
    gc.collect()

def file_generator(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                yield os.path.join(root, file)

def process_all_files(input_dir, output_dir, batch_size=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files_processed = 0
    for input_file in file_generator(input_dir):
        relative_path = os.path.relpath(input_file, input_dir)
        output_file_dir = os.path.join(output_dir, os.path.dirname(relative_path))
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
        process_audio_file(input_file, output_file_dir)
        files_processed += 1
        if files_processed % batch_size == 0:
            gc.collect()


process_all_files('resampled', 'Spettrogrammi', batch_size=50)