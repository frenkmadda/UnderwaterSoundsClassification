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


def calculate_ssim(image_pair,win_size=3):
    """
    Function tha actually calculate the ssim for a pair of images
    :param image_pair: the image pair to be compared
    :param win_size: the window size for the ssim calculation
    :return: the ssim score for the image pair
    """
    try:
        img1 = img_as_float(io.imread(image_pair[0], as_gray=True))
        img2 = img_as_float(io.imread(image_pair[1], as_gray=True))
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
        score = ssim(img1, img2,win_size=win_size, data_range= data_range)
        return {'Image1': image_pair[0], 'Image2': image_pair[1], 'SSIM': score}
    except Exception as e:
        logging.error(f"Error processing pair {image_pair}: {e}")
        return {'Image1': image_pair[0], 'Image2': image_pair[1], 'SSIM': None, 'Error': str(e)}


def calculate_ssim_for_image_pairs(csv_path, num_workers=5):
    """
    Function that calculate ssim for all the image pairs in a csv file containing the paths of the images
    it is optimized for parallel processing, the files are loaded in grayscale and the ssim is calculated
    the results are not influenced by the lack of colors although the performance gain is huge
    :param csv_path: the csv path containing the image paths
    :param num_workers: numero di workers per il parallel processing can be tuned as needed and depending on the machine
    :return: a csv files cotaning image pairs and their ssim
    """
    df = pd.read_csv(csv_path, header=None, names=['FilePath'])
    df['Subfolder'] = df['FilePath'].apply(lambda path: os.path.dirname(path))
    grouped_df = df.groupby('Subfolder')

    image_pairs = []
    for _, group in grouped_df:
        image_pairs.extend(combinations(group['FilePath'], 2))

    results = []

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for result in tqdm(executor.map(calculate_ssim, image_pairs), total=len(image_pairs)):
                results.append(result)
    except Exception as e:
        logging.error(f"Error in multiprocessing: {e}")

    results_df = pd.DataFrame(results)
    return results_df


ssim_results = calculate_ssim_for_image_pairs('Spettrogrammi/Target/df_paths_filtered.csv')
pd.set_option('display.max_rows', None)
ssim_results.to_csv('Spettrogrammi/Target/ssim_results.csv', index=False)



