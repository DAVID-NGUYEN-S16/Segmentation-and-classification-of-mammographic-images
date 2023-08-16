import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import os
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
import multiprocessing as mp
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from clahe import *
from haze_removal import *
def zero_mean_normalize(image):
    # Calculate mean value for the entire image
    mean_pixel = np.mean(image)

    # Subtract mean value from each pixel
    normalized_image = image - mean_pixel
    normalized_image[normalized_image < 0] = 0
    return np.uint8(normalized_image)
def preprocess_image(image_clahe, cl = 0, thred = 4, normalization = 0):
    # print(cl , thred , normalization )
    
    # if normalization == 1 : image_clahe = zero_mean_normalize(image_clahe)
    # if haze == 1:
        
    if cl == 1: 
        
        image_clahe = getRecoverScene(image_clahe, refine=True)
        image_clahe = cv2.cvtColor(np.uint8(image_clahe), cv2.COLOR_RGB2GRAY)
        image_clahe  = clahe(image_clahe ,thred,0,0)
    # image = getRecoverScene(image, refine=True)
    # print('0k')
    
    # print('0k')
    if len(image_clahe.shape) == 2:
        image_clahe = cv2.cvtColor(np.uint8(image_clahe), cv2.COLOR_GRAY2RGB)
    return image_clahe
def write_dict_to_txt(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            line = f"{key}: {value}\n"
            file.write(line)
def process(row, train = False, name = None, thred = 0, size = (224, 224), normal=0, path_global = 'Data_clahe/'):
    # row = .data.iloc[idx]

    theta_1 = 120
    theta_2 = 240
    # img_path = os.path.join(path_global, row["Image"])
    path = '/'.join([row.Image.split('/')[0], row.Image.split('/')[-1]])
    img_path = 'E:/Processing Data/No_apply_window/' + path
    img_path = img_path.replace('Data_image', 'Meta_data')
    print(img_path)
    image = cv2.imread(img_path)
    image = cv2.resize(image, size)
    height, width = image.shape[:2]
    
    # if train:
    if row['roatation_120'] == 1:
        rotation_matrix_1 = cv2.getRotationMatrix2D((width / 2, height / 2), theta_1, 1)
        image = cv2.warpAffine(image, rotation_matrix_1, (width, height))
    if row['roatation_240'] == 1:
        rotation_matrix_2 = cv2.getRotationMatrix2D((width / 2, height / 2), theta_2, 1)
        image = cv2.warpAffine(image, rotation_matrix_2, (width, height))
    # Apply zero-mean normalization
    # if train: 
    image = preprocess_image(image_clahe=image, cl = row['clahe'] , thred=thred, normalization=normal)
    # else: image = preprocess_image(image, cl = 0)
    if train: fl = 'Train'
    else: fl = 'Test'
    if row['status'] == 0:
        if os.path.exists(f'{path_global}{fl}/Bengin/') == False:
            os.makedirs(f'{path_global}{fl}/Bengin/')
        cv2.imwrite(f'{path_global}{fl}/Bengin/{name}.png', image)
    if row['status'] == 1:
        if os.path.exists(f'{path_global}{fl}/Mag/') == False:
            os.makedirs(f'{path_global}{fl}/Mag/')
        cv2.imwrite(f'{path_global}{fl}/Mag/{name}.png', image)
    
    # Description "E:\Processing Data\No_apply_window\Meta_data\mdb148.png"
    

def process_data(i, path_meta, path_global, name_dataset, normal, size, thred, train, title):
    meta = pd.read_csv(path_meta)
    row = meta.iloc[i]
    path = row.Image
    name = f"{os.path.basename(path).replace('.png', '')}_{row.roatation_120}_{row.roatation_240}_{row.clahe}"
    
    process(
        row=row, 
        train=train, 
        name=name, 
        thred=thred, 
        size=size, 
        normal=normal, 
        path_global=path_global,
    )

if __name__ == '__main__':
    path_meta = r"E:\Processing Data\NVN_LAM_LAI_TU_DAU\test_aug.csv"
    path_global = 'Data_clahe/'
    name_dataset = 'CIBS-DDSM'
    normal = 1
    size = (224, 224)
    thred = 2
    train = False
    title = 'Normalization after'

    print(path_meta)
    meta = pd.read_csv(path_meta)
    # with mp.Pool(5) as p:
    with mp.Pool(10) as p:
        partial_process_data = partial(
            process_data,
            path_meta=path_meta,
            path_global=path_global,
            name_dataset=name_dataset,
            normal=normal,
            size=size,
            thred=thred,
            train=train,
            title=title,
        )
        p.map(partial_process_data, list(meta.index))

    description_dict = {
        'Name Dataset': name_dataset,
        'Quantity Image': len(meta),
        'Image Size': size,
        'thred': thred,
        'Normalization': normal,
        'Detail': title,
        'Apply haze removal': 1
    }
    write_dict_to_txt(description_dict, path_global + f'description_{train}.txt')
