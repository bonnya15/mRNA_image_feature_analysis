# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:18:47 2024

@author: shiuli Subhra Ghosh
"""

import functools
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
import tifffile as tiff
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from scipy.spatial.distance import cdist
import src.nearest_neighbors as nn


# --- Decorator: log each single-image analysis call ---
def log_analysis(func):
    @functools.wraps(func)
    def wrapper(self, sol, num):
        print(f"[INFO] Analysing: solvent={sol}, image={num}")
        result = func(self, sol, num)
        print(f"[INFO] Done:      solvent={sol}, image={num}")
        return result
    return wrapper


# --- Decorator: loop over all solvents/numbers and collect results ---
def collect_results(func):
    @functools.wraps(func)
    def wrapper(self):
        for sol in self.solvents:
            alpha_list, neigh_list, mi_list, ma_list = [], [], [], []
            for num in self.numbers:
                alpha, neigh, mi, ma = func(self, sol, num)
                alpha_list.append(alpha)
                neigh_list.append(neigh)
                mi_list.append(mi)
                ma_list.append(ma)
            self.mean_alpha_list.append(alpha_list)
            self.mean_neigh_list.append(neigh_list)
            self.mean_mi_list.append(mi_list)
            self.mean_ma_list.append(ma_list)
    return wrapper


class ImageAnalysisPipeline:

    def __init__(
        self,
        base_path='./images',
        solvents=None,
        numbers=None,
        pixel_size_nm=(40.96 * 1e3) / 2560,
        max_distance_nm=20,
        neighbor_radius_nm=1000,
    ):
        self.base_path = base_path
        self.solvents = solvents or ["Water", "MeOH", "Acetone"]
        self.numbers = numbers or [1, 2, 3]
        self.pixel_size_nm = pixel_size_nm
        self.max_distance_pixels = max_distance_nm / self.pixel_size_nm
        self.neighbor_radius_pixels = neighbor_radius_nm / self.pixel_size_nm

        self.mean_alpha_list = []
        self.mean_neigh_list = []
        self.mean_mi_list = []
        self.mean_ma_list = []

    @log_analysis
    def _analyse_single(self, sol, num):
        img_path = f'{self.base_path}/{sol}/Averaged shifted histograms_{sol}-{num}.tif'
        img = tiff.imread(img_path)
        return nn.nearest_neighbors(img, self.max_distance_pixels, self.neighbor_radius_pixels, self.pixel_size_nm)

    @collect_results
    def _run_all(self, sol, num):
        return self._analyse_single(sol, num)

    def _build_dataframes(self):
        df_alpha = pd.DataFrame(np.array(self.mean_alpha_list).T, columns=self.solvents)
        df_neigh = pd.DataFrame(np.array(self.mean_neigh_list).T, columns=self.solvents)
        df_mi    = pd.DataFrame(np.array(self.mean_mi_list).T,    columns=self.solvents)
        df_ma    = pd.DataFrame(np.array(self.mean_ma_list).T,    columns=self.solvents)
        return df_alpha, df_neigh, df_mi, df_ma

    def run(self):
        self._run_all()
        return self._build_dataframes()

    def run_single(self, img_path):
        print(f"[INFO] Analysing single image: {img_path}")
        img = tiff.imread(img_path)
        alpha, neigh, mi, ma = nn.nearest_neighbors(img, self.max_distance_pixels, self.neighbor_radius_pixels, self.pixel_size_nm)
        print(f"[INFO] Done.")
        return {"alpha": alpha, "neighbours": neigh, "min_dist": mi, "max_dist": ma}


if __name__ == "__main__":

    pipeline = ImageAnalysisPipeline(
        base_path='./images',
        solvents=["Water", "MeOH", "Acetone"],
        numbers=[1, 2, 3],
    )

    # --- Run on all images ---
    # df_alpha, df_neigh, df_mi, df_ma = pipeline.run()

    # print("Alpha values are : \n", df_alpha)
    # print("Mean neighbouring cluster are : \n", df_neigh)
    # print("Mean minimum distance are : \n", df_mi)
    # print("Mean maximum distance values are : \n", df_ma)

    # --- Run on a single image ---

    pipeline = ImageAnalysisPipeline()
    alpha, neigh, mi, ma = pipeline._analyse_single("Water", 1)
    print(alpha, neigh, mi, ma)

    # result = pipeline.run_single(r'.\images\Water\Averaged shifted histograms_Water-1.tif')
    # print(result)
