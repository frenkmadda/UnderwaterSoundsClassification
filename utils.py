import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
def extractFrequencies(audiolist):
    frequencies = []
    for audio_file in audiolist:
        # Carica il file audio con librosa
        y, sr = librosa.load(audio_file, sr=None)
        frequencies.append(sr)
    return frequencies


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
            # Exclude .DS_Store and .csv files
            if file != '.DS_Store' and not file.endswith('.csv'):
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
        audio, frequenza_campionamento = librosa.load(percorso_file)

        durata = librosa.get_duration(y=audio, sr=frequenza_campionamento)
        audio_durations.append(durata)

    return audio_durations
