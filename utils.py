import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import soundfile as sf
from pydub.utils import mediainfo


def load_dataset(dataset_dir):
    dataset_dir = dataset_dir

    # Lista dei file audio trovati
    audio_files = []

    # Scansiona ricorsivamente la cartella del dataset
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            # Controlla se il file è un file audio (puoi aggiungere controlli per estensione)
            if file.endswith(".wav") or file.endswith(".mp3"):
                # Ottieni il percorso completo del file
                file_path = os.path.join(root, file)
                # Aggiungi il percorso del file alla lista dei file audio
                audio_files.append(file_path)

    return audio_files


#function to extract sampling rates given an audio lists
def get_frequencies(dfpath):
    df = pd.read_csv(dfpath)
    df = df[df['FilePath'].str.endswith(('.wav', '.mp3'))]

    frequencies_target = []
    frequencies_non_target = []

    for percorso_file in df['FilePath']:
        audio, sr = librosa.load(percorso_file, sr=None)

        if 'Non-Target' in percorso_file:
            frequencies_non_target.append(sr)
        else:
            frequencies_target.append(sr)

    return frequencies_target, frequencies_non_target



#Function to plot frequencies given two frequencies lists
def plotFrequencies(frequenciesTarget, frequenciesNonTarget):
    # Plotto separatemante le frequenze di target e non target
    plt.hist(frequenciesTarget, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Frequenza di Campionamento (Hz)')
    plt.ylabel('Numero di campioni')
    plt.title('Distribuzione delle Frequenze di Campionamento')
    # plt.xlim(xmin=0, xmax=80000)

    plt.hist(frequenciesNonTarget, bins='auto', color='red', alpha=0.7, rwidth=0.85)

    plt.show()


#function to create a dataframe from the audio datasets
def create_dataframe_from_files(dataset_dir):
    # Initialize two empty lists to store file paths and file names
    file_paths = []
    file_names = []

    # Walk through the directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            # Include only .mp3 and .wav files
            if file.endswith('.mp3') or file.endswith('.wav'):
                # Get the full file path
                file_path = os.path.join(root, file)
                # Append the file path and the file name to the respective lists
                file_paths.append(file_path)
                file_names.append(file)

    # Create a DataFrame from the list of file paths
    df_paths = pd.DataFrame(file_paths, columns=['FilePath'])
    # Create a DataFrame from the list of file names
    df_names = pd.DataFrame(file_names, columns=['FileName'])

    # Save the DataFrames to CSV files in the Dataset directory
    df_paths.to_csv(os.path.join(dataset_dir, 'df_paths.csv'), index=False)
    df_names.to_csv(os.path.join(dataset_dir, 'df_names.csv'), index=False)

    return df_paths, df_names


#Function to find and extract duplicated audios
def find_duplicates(df_names):
    # Reset the index of the DataFrame
    df_names = df_names.reset_index(drop=True)

    # Initialize an empty dictionary to store the duplicate file names and their indices
    duplicate_dict = {}

    # Find duplicate rows based on the 'FileName' column
    duplicates = df_names[df_names.duplicated(subset='FileName', keep=False)]

    # Group the duplicate rows by file name and get the indices for each group
    for file_name, group in duplicates.groupby('FileName'):
        # Add 2 to each index in the list of indices
        adjusted_indices = [index + 2 for index in group.index.tolist()]
        # Add the adjusted indices to the duplicate_dict
        duplicate_dict[file_name] = adjusted_indices

    return duplicate_dict

def remove_rows(df_paths, df_names, txt_file, paths_file, names_file):
    # Read the text file and split the paths
    with open(txt_file, 'r') as f:
        paths = f.read().split(',')

    if platform.system() == 'Windows':
        paths = [path.replace('/', '\\') for path in paths]
    else:
        paths = [path.replace('\\', '/') for path in paths]

    # Normalize the paths
    paths = [os.path.normpath(path) for path in paths]

    # Find the rows to remove
    rows_to_remove = df_paths['FilePath'].isin(paths)

    # Remove the rows from df_paths
    df_paths = df_paths[~rows_to_remove]

    # Remove the corresponding rows from df_names
    df_names = df_names[~rows_to_remove]

    # Save the updated DataFrames to CSV files
    df_paths.to_csv(paths_file, index=False)
    df_names.to_csv(names_file, index=False)

    return df_paths, df_names

#Function to extract audio durations
def extract_durations(dfpath):
    df = pd.read_csv(dfpath)

    df = df[df['FilePath'].str.endswith(('.wav', '.mp3'))]

    audio_durations = []

    for percorso_file in df['FilePath']:
        audio, frequenza_campionamento = librosa.load(percorso_file, sr=None)

        durata = librosa.get_duration(y=audio, sr=frequenza_campionamento)
        audio_durations.append(durata)

    return audio_durations

def plot_durations(audio_durations):
    max_duration = max(audio_durations)
    bins = np.arange(0, max_duration + 100, 100)

    # per aumentare il livello di dettaglio basta mettere bins=auto
    counts, bins, patches = plt.hist(audio_durations, bins=bins, color='blue', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Durata audio in secondi')
    plt.ylabel('Numero di campioni')
    plt.title('Distribuzione delle durate degli audio')

    # Aggiungi il numero di campioni su ogni classe
    for count, bin, patch in zip(counts, bins, patches):
        plt.text(bin, count, str(int(count)), color='black', ha='left', va='bottom',fontsize=8)

    plt.show()


# Function to extract audio files from a dataframe
def get_audio_files(dfpath):
    # Initialize an empty list to store the file paths
    audio_files = []

    df = pd.read_csv(dfpath)

    df = df[df['FilePath'].str.endswith(('.wav', '.mp3'))]

    # Traverse the directory
    for percorso_file in df['FilePath']:
        audio_files.append(percorso_file)

    return audio_files

# Function to analyze the max frequencies of audio files
def analyze_max_frequencies(audio_files):
    max_frequencies = []
    for file in audio_files:
        # Load the audio file
        y, sr = librosa.load(file, mono=True, sr=None)  # Ensure the audio is mono

        # Skip if the audio is too short
        if len(y) < 2048:
            print(f"Skipping {file} because it's too short")
            continue

        # Compute the spectral centroid frequencies
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)

        # Append the max frequency to the list
        max_frequencies.append(np.max(spectral_centroids))

    return max_frequencies

# Function to analyze the channels of audio files
def analyze_channels(audio_files):
    channels = []
    two_channel_files = []
    for file in audio_files:
        # Read the audio file
        data, samplerate = sf.read(file)

        # Get the number of channels (2 for stereo, 1 for mono)
        num_channels = 1 if len(data.shape) == 1 else data.shape[1]

        # Append the number of channels to the list
        channels.append(num_channels)

        # If the file has 2 channels, add it to the list
        if num_channels == 2:
            two_channel_files.append(file)

    # Print the paths of the files that have 2 channels
    print("Files with 2 channels:")
    for file in two_channel_files:
        print(file)
    
    return channels


# Function to plot the max frequencies of audio files
def plot_max_frequencies(max_frequencies):
    # Create a horizontal bar plot of the max frequencies
    plt.figure(figsize=(10, 30))

    counts, bins = np.histogram(max_frequencies, bins='auto')
    y = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]  # Calculate the y position for the text

    plt.barh(y, counts, height=np.diff(bins)*0.8, align='center', color='blue', alpha=0.7)  # Reduce the bar width by 50%

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Number of Audio Files')
    plt.ylabel('Max Frequency (Hz)')
    plt.title('Distribution of Max Frequencies')

    # Add the count of instances next to each bar, aligned vertically in the center
    for count, bin in zip(counts, y):
        if count > 0:  # Add the text only if the count is greater than 0
            plt.text(count, bin, str(int(count)), va='center', ha='left')

    plt.show()


# Function to plot the max frequencies of audio files using a boxplot
def boxplot_max_frequencies(max_frequencies):
    plt.figure(figsize=(5, 6))
    plt.boxplot(max_frequencies)
    plt.title('Boxplot of max frequencies')
    plt.ylabel('Frequency')
    plt.show()


# Plot the number of channels
def plot_channels(channels):
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(channels, bins=[1, 2, 3], align='left', rwidth=0.8)
    plt.title('Number of Channels')
    plt.xlabel('Channels')
    plt.ylabel('Count')
    plt.xticks([1, 2])

    # Add labels to the histogram bars
    for count, bin, patch in zip(counts, bins, patches):
        plt.text(bin, count, str(int(count)), color='black', ha='center', va='bottom')

    plt.show()


def plot_class_distribution(df, class_column):
    # Calcola la distribuzione delle classi
    class_distribution = df[class_column].value_counts()

    # Crea un plot a barre della distribuzione delle classi
    plt.figure(figsize=(30, 10))
    class_distribution.plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()


def plot_class_distribution_horizontal(df, class_column):
    # Calculate the class distribution
    class_distribution = df[class_column].value_counts()

    # Sort the class distribution in descending order
    class_distribution = class_distribution.sort_values(ascending=True)

    # Create a horizontal bar plot of the class distribution
    plt.figure(figsize=(10, 30))

    # Color the bars based on the class
    # bar_colors = ['red' if x == 'Target' else 'blue' for x in class_distribution.index]

    class_distribution.plot(kind='barh',  width=0.8)  # Increase the width of the bars

    # Add the count of instances next to each bar, aligned vertically in the center
    for index, value in enumerate(class_distribution):
        plt.text(value + 1, index, str(value), va='center')

    plt.xlabel('Count')
    plt.ylabel('Class')
    plt.title('Class Distribution')
    plt.show()



