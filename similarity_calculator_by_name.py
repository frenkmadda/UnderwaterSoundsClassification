import os
import pandas as pd
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from tqdm import tqdm
import logging
from skimage import io, img_as_float

# Set up logging
logging.basicConfig(level=logging.INFO, filename='ssim_debug.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_ssim(image_pair, win_size=3):
    """
    Function that actually calculates the SSIM for a pair of images
    :param image_pair: the image pair to be compared
    :param win_size: the window size for the SSIM calculation
    :return: the SSIM score for the image pair
    """
    try:
        img1 = img_as_float(io.imread(image_pair[0]))
        img2 = img_as_float(io.imread(image_pair[1]))
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
        score = ssim(img1, img2, win_size=win_size, data_range=data_range)
        return {'Image1': image_pair[0], 'Image2': image_pair[1], 'SSIM': score}
    except Exception as e:
        logging.error(f"Error processing pair {image_pair}: {e}")
        return {'Image1': image_pair[0], 'Image2': image_pair[1], 'SSIM': None, 'Error': str(e)}

def calculate_ssim_for_image_pairs(csv_path, num_workers=3):
    df = pd.read_csv(csv_path, header=None, names=['FilePath'])

    # Extract the file name from each path and remove the "_resampled_x.png" part
    df['FileName'] = df['FilePath'].apply(lambda path: os.path.basename(path).rsplit('_', 1)[0])

    # Group images by file name
    grouped_df = df.groupby('FileName')
    image_pairs = []
    for _, group in grouped_df:
        image_pairs.extend(combinations(group['FilePath'], 2))

    results = []

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            if image_pairs:
                for result in tqdm(executor.map(calculate_ssim, image_pairs), total=len(image_pairs)):
                    results.append(result)
    except Exception as e:
        logging.error(f"Error in multiprocessing: {e}")

    results_df = pd.DataFrame(results)

    return results_df

def read_csvs_from_folder(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    # Filter only CSV files whose name starts with 'df_paths'
    csv_files = [os.path.join(directory, file) for file in files if file.endswith('.csv') and file.startswith('df_paths')]
    # Return the list of full paths to the CSV files
    return csv_files

def main():
    csv_paths = read_csvs_from_folder('Spettrogrammi/Non-Target/dati ssim')

    output_dir = 'Spettrogrammi/Target/dati ssim/SSIM'
    os.makedirs(output_dir, exist_ok=True)

    for csv_path in csv_paths:
        # Get the file name from the path
        file_name = os.path.basename(csv_path)

        # Remove the file extension
        file_name_without_extension = os.path.splitext(file_name)[0]

        # Split the file name on the delimiter '_'
        parts = file_name_without_extension.split('_')

        # Take the desired part of the name (after 'df_paths_')
        name_part = parts[2]

        # Construct the output file path
        output_file_path = f'{output_dir}/ssim_results_{name_part}.csv'

        # If the output file already exists, skip this iteration
        if os.path.exists(output_file_path):
            continue

        ssim_results = calculate_ssim_for_image_pairs(csv_path)

        if not ssim_results.empty:
            pd.set_option('display.max_rows', None)
            ssim_results.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()
