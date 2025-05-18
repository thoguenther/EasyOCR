import easyocr
import os
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from itertools import product

# OCR-Reader initialisieren
reader = easyocr.Reader(['de'])

# Pfad zum Datensatz
dataset_path = "/home/tguenther/Data/GERALD_v1_5_thomas/cropped_images_speed_sorted/"

   

# Bildvorverarbeitung
# Erweiterte Bildvorverarbeitung mit Morphologie
def categorize_resolution(width):
    if width < 50:
        return 'niedrig'
    elif width <= 80:
        return 'mittel'
    else:
        return 'hoch'

# Bildvorverarbeitung (ohne binarize)
def preprocess_image(img,
                     blur=False, kernel_size=5,
                     clahe=False, clip_limit=2.0, tile_grid_size=(8, 8),
                     morph=False, morph_kernel_size=2, morph_iterations=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur:
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size <= 0:
            kernel_size = 3
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        gray = clahe_obj.apply(gray)

    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        gray = cv2.dilate(gray, kernel, iterations=morph_iterations)

    return gray

# Evaluation
def evaluate_combination(params, image_cache):
    results = []

    for item in image_cache:
        img_path = item['path']
        img = item['image']
        true_label = item['true_label']
        width = item['width']
        height = item['height']
        resolution_category = categorize_resolution(width)

        preprocessed = preprocess_image(
            img,
            blur=params['blur'],
            kernel_size=params['kernel_size'],
            clahe=params['clahe'],
            clip_limit=params['clip_limit'],
            tile_grid_size=params['tile_grid_size'],
            morph=params['morph'],
            morph_kernel_size=params['morph_kernel_size'],
            morph_iterations=params['morph_iterations']
        )

        result = reader.readtext(preprocessed, detail=0)
        predicted_label = int(result[0]) if result and result[0].isdigit() else None
        correct = predicted_label == true_label

        results.append({
            'path': img_path,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': correct,
            'width': width,
            'height': height,
            'resolution_category': resolution_category
        })

    df = pd.DataFrame(results)
    accuracy_by_cat = df.groupby('resolution_category')['correct'].mean().to_dict()
    try:
        mae = mean_absolute_error(
            df.dropna(subset=['predicted_label'])['true_label'],
            df.dropna(subset=['predicted_label'])['predicted_label']
        )
    except:
        mae = None
    return df, accuracy_by_cat, mae

# Hauptfunktion: Grid-Search ohne Binarisierung, mit erweitertem Morphologie-Tuning
def run_grid_search_morph_only():
    blur_opts = [False, True]
    kernel_sizes = [5, 7, 9]

    clahe_opts = [False, True]
    clip_limits = [2.0]
    tile_grid_sizes = [(8, 8)]

    morph_opts = [False, True]
    morph_kernel_sizes = [1, 2, 3]
    morph_iterations = [1, 2]

    param_grid = [
        {'blur': bl, 'kernel_size': ks,
         'clahe': cl, 'clip_limit': climit, 'tile_grid_size': tg,
         'morph': m, 'morph_kernel_size': mks, 'morph_iterations': mi}
        for bl in blur_opts
        for ks in kernel_sizes
        for cl in clahe_opts
        for climit in clip_limits
        for tg in tile_grid_sizes
        for m in morph_opts
        for mks in morph_kernel_sizes
        for mi in morph_iterations
    ]

    # Bilder cachen
    image_cache = []
    for folder_name in sorted(os.listdir(dataset_path)):
        if not folder_name.isdigit():
            continue
        true_label = int(folder_name.lstrip('0'))
        folder_path = os.path.join(dataset_path, folder_name)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            image_cache.append({
                'path': img_path,
                'image': img,
                'true_label': true_label,
                'width': w,
                'height': h
            })

    summary = []
    best_combo = None
    best_accuracy_hoch = -1
    best_df = None

    for params in tqdm(param_grid, desc="Grid-Search mit Morphologie (ohne Binarisierung)"):
        df, accuracy, mae = evaluate_combination(params, image_cache)
        acc_hoch = accuracy.get('hoch', 0)

        summary.append({
            **params,
            'accuracy_hoch': acc_hoch,
            'accuracy_mittel': accuracy.get('mittel', 0),
            'accuracy_niedrig': accuracy.get('niedrig', 0),
            'mae': mae
        })

        if acc_hoch > best_accuracy_hoch:
            best_accuracy_hoch = acc_hoch
            best_combo = params
            best_df = df

    df_summary = pd.DataFrame(summary)
    return best_df, df_summary, best_combo

