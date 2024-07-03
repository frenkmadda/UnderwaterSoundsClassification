import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from tkinter import Image

def load_spectrogram(image_path):
    """
    Load a spectrogram image and return it as a numpy array.

    :param image_path: Path to the spectrogram image.

    :return: Tuple containing the spectrogram as a numpy array and the image mode.
    """
    image = Image.open(image_path)
    spectrogram = np.array(image) / 255
    return spectrogram, image.mode

def time_shift_spectrogram(image_path, output_path, i, duration=3):
    """
    Perform a time shift on a spectrogram image.

    :param image_path: Path to the input spectrogram image.
    :param output_path: Path to save the shifted spectrogram image.
    :param i: Index to determine the shift amount.
    :param duration: Duration of the audio in seconds.

    :return: None
    """
    image = Image.open(image_path)

    image_array = np.array(image, dtype=np.uint8)

    shift_seconds = [0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72,
                     0.80, 0.88, 0.96, 1.04, 1.12, 1.20, 1.28, 1.36, 1.44, 1.52]

    width = image_array.shape[1]
    pixels_per_second = width // duration
    shift_pixels = int(pixels_per_second * shift_seconds[i])

    shifted_image_array = np.zeros_like(image_array, dtype=np.uint8)

    shifted_image_array[:, shift_pixels:] = image_array[:, :-shift_pixels]

    silence = np.zeros((image_array.shape[0], shift_pixels, image_array.shape[2]), dtype=np.uint8)
    silence[:, :, 3] = 255  # Set the alpha channel to 255 for full opacity

    shifted_image_array[:, :shift_pixels] = silence

    shifted_image = Image.fromarray(shifted_image_array)

    shifted_image.save(output_path)


def time_shift(image_path, save_dir, i):
    """
    Apply time shift to an image and save it.

    :param image_path: Path to the input spectrogram image.
    :param save_dir: Directory to save the shifted image.
    :param i: Index to determine the shift amount.

    :return: Tuple containing the path to the shifted image and a flag indicating if the operation was successful.
    """
    shifted_image_path = os.path.join(save_dir, f'shifted_{i}_{os.path.basename(image_path)}')

    if not os.path.exists(shifted_image_path):
        time_shift_spectrogram(image_path, shifted_image_path, i)
        os.makedirs(save_dir, exist_ok=True)
        return shifted_image_path, True
    else:
        return "", False


def add_white_noise(spectrogram, mode, save_path, i):
    """
    Add white noise to a spectrogram image.

    :param spectrogram: Numpy array representing the spectrogram.
    :param mode: Image mode.
    :param save_path: Path to save the noisy spectrogram image.
    :param i: Index to determine the noise factor.

    :return: None
    """
    noise_factor = [0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.0020,
                    0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029]

    noise = np.random.randn(*spectrogram.shape) * noise_factor[
        i]  # Modificare sta riga se necessario per avere più opzioni di rumore (ad esempio fare /2, ecc....)
    augmented_spectrogram = spectrogram + noise
    np.clip(augmented_spectrogram, 0, 1)  # Clippa i valori tra 0 e 1

    augmented_spectrogram = (augmented_spectrogram * 255).astype(np.uint8)  # Converti i valori a 0-255
    image = Image.fromarray(augmented_spectrogram, mode)  # Converti in immagine mantenendo il mode originale
    image.save(save_path)


def white_noise(image_path, save_dir, i):
    """
    Add white noise to an image and save it.

    :param image_path: Path to the input spectrogram image.
    :param save_dir: Directory to save the noisy image.
    :param i: Index to determine the noise factor.

    :return: Flag indicating if the operation was successful.
    """
    spectrogram, mode = load_spectrogram(image_path)

    noisy_image_path = os.path.join(save_dir, f'noisy_{i}_{os.path.basename(image_path)}')

    if not os.path.exists(noisy_image_path):
        add_white_noise(spectrogram, mode, noisy_image_path, i)
        os.makedirs(save_dir, exist_ok=True)
        return True
    else:
        return False


def increase_and_decrease_volume(spectrogram, save_path, mode, i, type):
    """
    Increase or decrease the volume of a spectrogram image.

    :param spectrogram: Numpy array representing the spectrogram.
    :param save_path: Path to save the modified spectrogram image.
    :param mode: Image mode.
    :param i: Index to determine the volume factor.
    :param type: Type of volume adjustment ('i' for increase, 'd' for decrease).

    :return: None
    """
    if type == 'i':
        volume_factor = [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5,
                         1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
    else:  # type == 'd'
        volume_factor = [0.15, 0.175, 0.19, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                         0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    spectrogram[..., :3] *= volume_factor[i]

    spectrogram = np.clip(spectrogram, 0, 255)

    amplified_image = Image.fromarray(spectrogram.astype(np.uint8), mode)
    amplified_image.save(save_path)


def increase_and_decrease(image_path, save_dir, i, type):
    """
    Increase or decrease the volume of an image and save it.

    :param image_path: Path to the input spectrogram image.
    :param save_dir: Directory to save the modified image.
    :param i: Index to determine the volume factor.
    :param type: Type of volume adjustment ('i' for increase, 'd' for decrease).

    :return: Flag indicating if the operation was successful.
    """
    image = Image.open(image_path)
    spectrogram = np.array(image).astype(np.float32)
    mode = image.mode

    if type == 'i':
        increase_image_path = os.path.join(save_dir, f'increase_{i}_{os.path.basename(image_path)}')
    else:  # type == 'd'
        increase_image_path = os.path.join(save_dir, f'decrease_{i}_{os.path.basename(image_path)}')

    if os.path.exists(increase_image_path):
        return False
    else:
        increase_and_decrease_volume(spectrogram, increase_image_path, mode, i, type)

        os.makedirs(save_dir, exist_ok=True)
        return True

def process_csv(csv_path, save_dir, value_counts):
    """
    Process a CSV file containing image paths, performing data augmentation on each image.

    :param csv_path: Path to the CSV file.
    :param save_dir: Directory to save the augmented images.
    :param value_counts: Pandas Series with counts of each class.

    :return: None
    """
    df = pd.read_csv(csv_path)
    max = np.max(value_counts)
    i = 0

    while np.any(value_counts < max):
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
            image_path = row.iloc[0]

            parts = image_path.split('/')
            relevant_parts = parts[1:-1]

            destination_dir = os.path.join(save_dir, *relevant_parts[:-1])
            destination_path = os.path.join(destination_dir, relevant_parts[-1])

            os.makedirs(destination_path, exist_ok=True)
            # If the curent class has reached the maximum, go to the next iteration
            if value_counts.loc[parts[2]] >= max:
                continue

            shifted_path, done_f = time_shift(image_path, destination_path, i)
            if done_f:
                value_counts.loc[parts[2]] += 1

            # Portion of the code to merge time shift and increase and decrease, done_f is a flag that indicates if the time shift has been done to then also do the increase and decrease, if there was no done_f in case of duplicate, the function would crash
            if value_counts.loc[parts[2]] < max and done_f:
                done = increase_and_decrease(shifted_path, destination_path, i, 'i')
                if done:
                    value_counts.loc[parts[2]] += 1

            if value_counts.loc[parts[2]] < max:
                done = white_noise(image_path, destination_path, i)
                if done:
                    value_counts.loc[parts[2]] += 1

            if value_counts.loc[parts[2]] < max:
                done = increase_and_decrease(image_path, destination_path, i, 'i')
                if done:
                    value_counts.loc[parts[2]] += 1

            if value_counts.loc[parts[2]] < max:
                done = increase_and_decrease(image_path, destination_path, i, 'd')
                if done:
                    value_counts.loc[parts[2]] += 1

            i += 1
            if i == 19:
                i = 0

def add_rows_to_csv(root_dir, csv_path, new_csv_path):
    """
    Adda new rows to an existing CSV file containing image paths.

    :param root_dir: The root directory to search for .png files.
    :param csv_path: The path to the existing CSV file to which new rows will be added.
    :param new_csv_path: The path where the updated CSV file will be saved.

    :return: None
    """
    df = pd.read_csv(csv_path)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                full_path = os.path.join(dirpath, filename)

                class_name = os.path.basename(dirpath)

                file_name = full_path.split('_resampled')[0]

                new_row = pd.DataFrame({'FilePath': [full_path], 'Classe': [class_name], 'File': [file_name], 'Set': ['train']})

                df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(new_csv_path, index=False)

def merge_csv_files(csv1_path, csv2_path, output_path):
    """
    Merges two CSV files into one and shuffles the rows.

    :param csv1_path: Path to the first CSV file.
    :param csv2_path: Path to the second CSV file.
    :param output_path: Path where the merged and shuffled CSV file will be saved.

    This function reads two CSV files, concatenates them, assigns a 'Label' based on the presence of 'Non-Target' in the 'FilePath',
    shuffles the rows randomly, and saves the result to `output_path`.
    """
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    df = pd.concat([df1, df2], ignore_index=True)

    df['Label'] = df['FilePath'].apply(lambda x: 'Non-Target' if "Non-Target" in x else 'Target')

    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(output_path, index=False)


def process_csv_merge(csv_path, save_dir, value_counts):
    """
    Processes a CSV file to balance classes through data augmentation.

    :param csv_path: Path to the CSV file containing image paths.
    :param save_dir: Directory where augmented images will be saved.
    :param value_counts: Pandas Series with counts of each class to balance.

    This function iterates over each row in the CSV file, performing data augmentation on images until all classes are balanced.
    It uses the 'increase_and_decrease' function to augment images by increasing or decreasing their volume based on the 'shifted' status and 'increase' absence in the file path.
    The process continues until the count of images in each class matches the maximum count found in `value_counts`.
    """
    df = pd.read_csv(csv_path)
    max = np.max(value_counts)
    i = 0

    while np.any(value_counts < max):
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
            image_path = row.iloc[0]

            parts = image_path.split('/')
            if 'Spettrogrammi' in parts[0]:
                relevant_parts = parts[1:-1]
                label = parts[1]
            else:
                relevant_parts = parts[2:-1]
                label = parts[2]

            destination_dir = os.path.join(save_dir, *relevant_parts[:-1])
            destination_path = os.path.join(destination_dir, relevant_parts[-1])

            os.makedirs(destination_path, exist_ok=True)
            # If the curent class has reached the maximum, go to the next iteration
            if value_counts.loc[label] >= max:
                continue

            if value_counts.loc[label] < max and "shifted" in image_path and "increase" not in image_path:
                done = increase_and_decrease(image_path, destination_path, i, 'd')
                if done:
                    value_counts.loc[parts[2]] += 1

            i += 1
            if i == 19:
                i = 0

def count_files_in_folder(folder_path):
    """
    Count the number of files in a folder.

    :param folder_path: Path to the folder.

    :return: Number of files in the folder.
    """
    count = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.png'):
                count += 1
    return count