import cv2
import pydicom
import numpy as np
from pydicom.pixel_data_handlers import apply_windowing
import imageio
import multiprocessing as mp
import glob
import os

from scipy import stats as sst
from scipy import stats as sst
import pandas as pd
meta = pd.read_csv(r"E:\BB\meta.csv")
def process1(i):
    img = cv2.imread(f'E:\WORKBASE\Project-rsna-breast-cancer-detection\Low energy images of CDD-CESM\{meta.Image_name[i]}.jpg')
    # print(f'E:\WORKBASE\Project-rsna-breast-cancer-detection\Low energy images of CDD-CESM\{meta.Image_name[i]}.jpg')
    if os.path.exists(f'E:\WORKBASE\Project-rsna-breast-cancer-detection\Low energy images of CDD-CESM\{meta.Image_name[i]}.jpg'):
        # print(f'E:\WORKBASE\Project-rsna-breast-cancer-detection\Low energy images of CDD-CESM\{meta.Image_name[i]}.jpg')
        imageio.imwrite(f"E:\BB\{meta.Image_name[i]}.png", img)
  
def my_method(ls):
    with mp.Pool(12) as p:
        p.map(process1, ls)