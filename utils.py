import os
import wave
import gc
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import mediainfo
import csv
from collections import defaultdict


def load_dataset(dataset_dir):
    '''
    Load the dataset from the given directory
    :param dataset_dir: the directory of the dataset
    :return: the folder with the audio files
    '''
    dataset_dir = dataset_dir

    audio_files = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)

    return audio_files


def get_frequencies(dfpath):
    """
    Get the frequencies of the audio files
    :param dfpath: the csv paths file
    :return:
    """
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


def plot_frequencies(frequenciesTarget, frequenciesNonTarget):
    """
    Plot the frequencies of the audio files
    :param frequenciesTarget: target frequencies
    :param frequenciesNonTarget: non target frequencies
    """
    plt.hist(frequenciesTarget, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Frequenza di Campionamento (Hz)')
    plt.ylabel('Numero di campioni')
    plt.title('Distribuzione delle Frequenze di Campionamento')
    # plt.xlim(xmin=0, xmax=80000)

    plt.hist(frequenciesNonTarget, bins='auto', color='red', alpha=0.7, rwidth=0.85)

    plt.show()


def create_dataframe_from_files(dataset_dir):
    """
    Create a DataFrame from the files in the given directory
    :param dataset_dir: the directory of the dataset
    :return: the DataFrame with the file paths and the DataFrame with the file names
    """
    file_paths = []
    file_names = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.mp3') or file.endswith('.wav') or file.endswith('.png'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                file_names.append(file)

    df_paths = pd.DataFrame(file_paths, columns=['FilePath'])
    df_names = pd.DataFrame(file_names, columns=['FileName'])

    df_paths.to_csv(os.path.join(dataset_dir, 'df_paths.csv'), index=False)
    df_names.to_csv(os.path.join(dataset_dir, 'df_names.csv'), index=False)

    return df_paths, df_names


def find_duplicates(df_names):
    """
    Find duplicate file names in the DataFrame and return a dictionary with the duplicate file names and their indices.
    :param df_names: dataframe with the file names
    :return: dictionary with the duplicate file names and their indices
    """
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
    """
    Remove rows from the DataFrames based on the file paths in the text file and save the updated DataFrames to CSV files.
    :param df_paths: dataframe with the file paths
    :param df_names: dataframe with the file names
    :param txt_file:
    :param paths_file:
    :param names_file:
    :return: the new DataFrames
    """
    with open(txt_file, 'r') as f:
        paths = f.read().split(',')

    if platform.system() == 'Windows':
        paths = [path.replace('/', '\\') for path in paths]
    else:
        paths = [path.replace('\\', '/') for path in paths]

    paths = [os.path.normpath(path) for path in paths]

    rows_to_remove = df_paths['FilePath'].isin(paths)

    df_paths = df_paths[~rows_to_remove]

    df_names = df_names[~rows_to_remove]

    df_paths.to_csv(paths_file, index=False)
    df_names.to_csv(names_file, index=False)

    return df_paths, df_names


def extract_durations(dfpath):
    """
    Extract the durations of audio files from a DataFrame.
    :param dfpath: the dataframe path
    :return: list of audio durations
    """
    df = pd.read_csv(dfpath)

    df = df[df['FilePath'].str.endswith(('.wav', '.mp3'))]

    audio_durations = []

    for percorso_file in df['FilePath']:
        audio, frequenza_campionamento = librosa.load(percorso_file, sr=None)

        durata = librosa.get_duration(y=audio, sr=frequenza_campionamento)
        audio_durations.append(durata)

    return audio_durations


def plot_durations(audio_durations):
    """
    Plot the distribution of audio durations.
    :param audio_durations: list of audio durations
    :return: none
    """
    max_duration = max(audio_durations)
    bins = np.arange(0, max_duration + 100, 100)

    counts, bins, patches = plt.hist(audio_durations, bins=bins, color='blue', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Durata audio in secondi')
    plt.ylabel('Numero di campioni')
    plt.title('Distribuzione delle durate degli audio')

    for count, bin, patch in zip(counts, bins, patches):
        plt.text(bin, count, str(int(count)), color='black', ha='left', va='bottom',fontsize=8)

    plt.show()


def get_audio_files(dfpath):
    """
    Get the audio files from the DataFrame
    :param dfpath: the dataframe with the audio files paths
    :return: the list of audio files
    """
    audio_files = []

    df = pd.read_csv(dfpath)

    df = df[df['FilePath'].str.endswith(('.wav', '.mp3'))]

    for percorso_file in df['FilePath']:
        audio_files.append(percorso_file)

    return audio_files


def analyze_max_frequencies(audio_files):
    """
    Function to extract the max frequencies from the audio files
    :param audio_files: audio files
    :return: a list with max frequencies
    """

    max_frequencies = []
    for file in audio_files:
        y, sr = librosa.load(file, mono=True, sr=None)

        if len(y) < 2048:
            print(f"Skipping {file} because it's too short")
            continue

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)

        max_frequencies.append(np.max(spectral_centroids))

    return max_frequencies


def analyze_channels(audio_files):
    """
    Function to analyze the number of channels of the audio files
    :param audio_files:
    :return:
    """

    channels = []
    two_channel_files = []
    for file in audio_files:
        data, samplerate = sf.read(file)

        num_channels = 1 if len(data.shape) == 1 else data.shape[1]

        channels.append(num_channels)

        if num_channels == 2:
            two_channel_files.append(file)

    print("Files with 2 channels:")
    for file in two_channel_files:
        print(file)
    
    return channels


def plot_max_frequencies(max_frequencies):
    """
    Plot the distribution of max frequencies
    :param max_frequencies: list of max frequencies
    """
    plt.figure(figsize=(10, 30))

    counts, bins = np.histogram(max_frequencies, bins='auto')
    y = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

    plt.barh(y, counts, height=np.diff(bins)*0.8, align='center', color='blue', alpha=0.7)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Number of Audio Files')
    plt.ylabel('Max Frequency (Hz)')
    plt.title('Distribution of Max Frequencies')

    for count, bin in zip(counts, y):
        if count > 0:
            plt.text(count, bin, str(int(count)), va='center', ha='left')

    plt.show()


def boxplot_max_frequencies(max_frequencies):
    """
    Boxplot of max frequencies
    :param max_frequencies:
    :return:
    """
    plt.figure(figsize=(5, 6))
    plt.boxplot(max_frequencies)
    plt.title('Boxplot of max frequencies')
    plt.ylabel('Frequency')
    plt.show()


# Plot the number of channels
def plot_channels(channels):
    """
    Plot the number of channels
    :param channels: the list of channels of all the audios
    :return:
    """
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(channels, bins=[1, 2, 3], align='left', rwidth=0.8)
    plt.title('Number of Channels')
    plt.xlabel('Channels')
    plt.ylabel('Count')
    plt.xticks([1, 2])

    for count, bin, patch in zip(counts, bins, patches):
        plt.text(bin, count, str(int(count)), color='black', ha='center', va='bottom')

    plt.show()


def plot_class_distribution(df, class_column):
    """
    Plot the class distribution of the DataFrame
    :param df: the dataframe
    :param class_column: class column
    :return:
    """
    class_distribution = df[class_column].value_counts()

    plt.figure(figsize=(30, 10))
    class_distribution.plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()


def plot_class_distribution_horizontal(df, class_column):
    """
    Plot the class distribution of the DataFrame horizontally
    :param df:
    :param class_column:
    :return:
    """
    class_distribution = df[class_column].value_counts()

    class_distribution = class_distribution.sort_values(ascending=True)

    plt.figure(figsize=(10, 30))
    class_distribution.plot(kind='barh',  width=0.8)

    for index, value in enumerate(class_distribution):
        plt.text(value + 1, index, str(value), va='center')

    plt.xlabel('Count')
    plt.ylabel('Class')
    plt.title('Class Distribution')
    plt.show()


def convert_mp3_to_wav(input_file, output_file):
    """
    Convert an mp3 file to wav
    :param input_file: file mp3 in input
    :param output_file: file wav in output
    """
    audio = AudioSegment.from_mp3(input_file)
    audio.export(output_file, format="wav")


def analyze_and_plot_bit_depth(audio_files):
    """
    Analyze and plot the bit depth distribution of the audio files
    :param audio_files: the audio files
    :return:
    """
    bit_depths = []
    countskipped= 0
    totalChampions = 0
    for file in audio_files:
        if file.lower().endswith('.mp3'):  # Ignora i file .mp3
            continue
        try:
            with wave.open(file, 'rb') as wf:
                bit_depth = wf.getsampwidth() * 8  # Convert sample width to bit depth
                bit_depths.append(bit_depth)
                totalChampions = totalChampions + 1
        except:
            print(f"Could not process file {file}")
            countskipped = countskipped + 1
            continue

    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(bit_depths, bins=30, edgecolor='black')
    plt.xlabel('Bit Depth')
    plt.ylabel('Count')
    plt.title('Bit Depth Distribution')

    bin_width = bins[1] - bins[0]  # Calculate the bin width

    for count, bin, patch in zip(counts, bins, patches):
        height = patch.get_height()
        if count > 0:
            plt.text(bin + bin_width/2, height * 1.01, str(int(count)), fontsize=12, ha='center')

    plt.show()

    print("Totale skippati: ", countskipped)
    print("Totale campioni analizzati: ", totalChampions)


def resample_and_split_audio_files(audio_files, target_sr, split_duration, output_dir='resampled'):
    """
    Resample, split and make the audio files mono
    :param audio_files: the audio files
    :param target_sr: the target sample rate
    :param split_duration: the duration of a splitted audio
    :param output_dir: the new directory for the resampled audio files
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in audio_files:
        # Check if the file is mp3 and the OS is macOS
        if file.lower().endswith('.mp3') and platform.system() == 'Darwin':
            # Convert mp3 to wav
            wav_file = file.replace('.mp3', '.wav')
            convert_mp3_to_wav(file, wav_file)
            file_to_load = wav_file
        else:
            file_to_load = file

        y, sr = librosa.load(file_to_load, sr=None, mono=True)

        # If we created a temporary wav file, delete it
        if file_to_load != file:
            os.remove(file_to_load)

        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        relative_path = os.path.relpath(file, 'Dataset')
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        base_name = os.path.splitext(os.path.basename(file))[0]
        frame_length = int(split_duration * target_sr)

        for i in range(0, len(y_resampled), frame_length):
            frame = y_resampled[i:i+frame_length]

            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))

            output_file = os.path.join(output_subdir, f'{base_name}_resampled_{i // frame_length}.wav')
            sf.write(output_file, frame, target_sr)


def create_csv_for_folders(directory, k):
    """
    Create a csv file for each folder in the given directory
    :param directory:
    :param k:
    :return:
    """
    data_dir = os.path.join(directory, 'dati ssim')
    os.makedirs(data_dir, exist_ok=True)

    for root, dirs, files in os.walk(directory):
        png_files = [file for file in files if file.endswith('.png')]
        if len(png_files) < k:
            continue
        df = pd.DataFrame([os.path.join(root, file) for file in png_files], columns=['FilePath'])
        folder_name = os.path.basename(root)
        df.to_csv(os.path.join(data_dir, f'df_paths_{folder_name}.csv'), index=False)


def merge_csv_files(directory, output_file):
    """
    Merge the csv files in the given directory
    :param directory:
    :param output_file:
    :return:
    """
    files = os.listdir(directory)

    csv_files = [file for file in files if file.endswith('.csv')]

    df_list = []

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(directory, csv_file))
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)


def filter_and_count(csv_path):
    """
    Filter the DataFrame by SSIM >= 0.95 and count the occurrences of each Image1 path.
    :param csv_path: the path of the csv file
    :return: the path of the Image with the most occurrences
    """
    df = pd.read_csv(csv_path)
    filtered_df = df[df['SSIM'] >= 0.95]
    count_df = filtered_df['Image1'].value_counts().reset_index()
    count_df.columns = ['Image1', 'Count']
    sorted_df = count_df.sort_values(by='Count', ascending=False)

    if sorted_df.empty:
        return None

    return sorted_df.iloc[0]['Image1']


def get_image2_paths(ssim_csv_path, image1_path):
    """
    Get all the paths of the images that are related to the image1_path
    :param ssim_csv_path: path to ssim_results.csv
    :param image1_path: path of the image to search
    :return: list of image2 paths related to image1_path
    """
    df = pd.read_csv(ssim_csv_path)
    similar_images_df = df[df['Image1'] == image1_path]
    similar_images_list = similar_images_df['Image2'].tolist()
    return similar_images_list


def remove_similar_images(ssim_csv_path, df_paths_csv_path, class_column):
    """
    Remove similar images from the dataset
    :param ssim_csv_path: path of csv ssim_results.csv
    :param df_paths_csv_path: df_paths
    :param class_column: list of classes
    """

    while True:
        isGood = False

        # Ottieni il percorso dell'immagine con il maggior numero di occorrenze in ssim_results.csv con SSIM >= 0.95
        image1_path = filter_and_count(ssim_csv_path)
        list_image2 = get_image2_paths(ssim_csv_path, image1_path)
        ssim_df = pd.read_csv(ssim_csv_path)
        ssim_df = ssim_df[ssim_df['Image1'] != image1_path]

        ssim_df.to_csv(ssim_csv_path, index=False)

        # Se filter_and_count restituisce None
        if image1_path is None:
            break

        path_parts = image1_path.split('/')
        if path_parts[2] in class_column:
            isGood = True

        if isGood:
            df_paths = pd.read_csv(df_paths_csv_path)

            for path in list_image2:
                df_paths = df_paths[df_paths['FilePath'] != path]

            df_paths.to_csv(df_paths_csv_path, index=False)

        else:
            continue


def get_class_distribution_over(df, class_column):
    '''
    Given a DataFrame, return the distribution of the classes with more than 500 images
    :param df: the dataframe
    :param class_column: class column
    :return: the class distribuition
    '''
    class_distribution = df[class_column].value_counts() > 500
    class_distribution = class_distribution.sort_values(ascending=True)
    return class_distribution


def remove_rows_by_class(ssim_csv_path, class_column):
    '''
    Remove rows from ssim_results.csv that belong to classes with less than 500 images to ease the removal of similar images
    :param ssim_csv_path: the ssim csv path
    :param class_column: the class list
    '''
    ssim_df = pd.read_csv(ssim_csv_path)

    # Create a new column 'class' in the DataFrame by extracting the class from 'Image1'
    ssim_df['class'] = ssim_df['Image1'].apply(lambda x: x.split('/')[2])

    # Create a condition that checks if 'class' is in 'class_column'
    condition = ssim_df['class'].isin(class_column)

    # Use the condition to filter the DataFrame
    filtered_df = ssim_df[condition]

    # Save the filtered DataFrame to the original file
    filtered_df.to_csv(ssim_csv_path, index=False)


def estrai_info_da_csv(input_csv, output_csv):
    '''
    Extract every class and the number of distinct audio associated with it
    :param input_csv: csv file to read
    :param output_csv:
    :return:
    '''
    classi = defaultdict(set)
    with open(input_csv, mode='r', newline='') as file:
        reader = csv.reader(file)
        first_row = next(reader)
        if first_row[0] == 'FilePath':
            first_row = next(reader)
        for row in reader:
            path = row[0]
            if platform.system() == 'Windows':
                classe = path.split(os.sep)[1]
                audio = path.split(os.sep)[2].split('_resampled_')[0]
            elif platform.system() == 'Darwin':
                classe = path.split(os.sep)[2]
                audio = path.split(os.sep)[3].split('_resampled_')[0]
            classi[classe].add(audio)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Classe', 'Numero Audio associato alla classe'])
        for classe, audio_set in classi.items():
            writer.writerow([classe, len(audio_set)])


def filter_csv(class_num_csv, csv_to_filter, output_di_file):
    """
    Filter the csv file by the number of audio for each class using class_num_csv and removing the folder paths from csv_to_filter
    the result is written in a new csv file given it's path in output_di_file
    :param class_num_csv: the csv containing the number of distinct audio for each class
    :param csv_to_filter: the source csv file to filter
    :param output_di_file: the path for the output csv file
    """
    df = pd.read_csv(class_num_csv)
    df_filtered = df[df['Numero Audio associato alla classe'] >= 10]

    df_paths = pd.read_csv(csv_to_filter)
    if platform.system() == 'Windows':
        df_paths['Classe'] = df_paths['FilePath'].apply(lambda x: x.split('\\')[1])
    elif platform.system() == 'Darwin':
        df_paths['Classe'] = df_paths['FilePath'].apply(lambda x: x.split('/')[2])
    df_paths_filtered = df_paths[df_paths['Classe'].isin(df_filtered['Classe'])]
    df_paths_filtered.to_csv(output_di_file, index=False)