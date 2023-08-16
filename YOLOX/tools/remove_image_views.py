import cv2
import imageio
import pandas as pd
import os
import multiprocessing as mp
class move_image:
    def __init__(self, data, stores = None, crop = True):
        self.meta = data
        self.stores = stores
        self.crop = crop
        
    def xx(self, id):
        par = self.meta.patient_id[id]
        lat = self.meta.laterality[id]
        view = self.meta.view[id]
        file_name = f"D:/OneDrive - Industrial University of HoChiMinh City/WORKBASE/Project-rsna-breast-cancer-detection/Data_main/Dataset_VIEW/{par}/{view}/{lat}/{self.meta.view[id]}_{self.meta.laterality[id]}_{self.meta.image_id[id]}.png"
        img = cv2.imread(file_name)
        # if self.crop:
        #     img = process_crop(img)
        
        # if os.path.exists(f"D:/OneDrive - Industrial University of HoChiMinh City/WORKBASE/Project-rsna-breast-cancer-detection/Data_main/Dataset_VIEW/{par}/") == False:
        #     os.mkdir(f"D:/OneDrive - Industrial University of HoChiMinh City/WORKBASE/Project-rsna-breast-cancer-detection/Data_main/Dataset_VIEW/{par}/")
        # if os.path.exists(f"D:/OneDrive - Industrial University of HoChiMinh City/WORKBASE/Project-rsna-breast-cancer-detection/Data_main/Dataset_VIEW/{par}/{view}/") == False:
        #     os.mkdir(f"D:/OneDrive - Industrial University of HoChiMinh City/WORKBASE/Project-rsna-breast-cancer-detection/Data_main/Dataset_VIEW/{par}/{view}/")
        # if os.path.exists(f"D:/OneDrive - Industrial University of HoChiMinh City/WORKBASE/Project-rsna-breast-cancer-detection/Data_main/Dataset_VIEW/{par}/{view}/{lat}/") == False:
        #     os.mkdir(f"D:/OneDrive - Industrial University of HoChiMinh City/WORKBASE/Project-rsna-breast-cancer-detection/Data_main/Dataset_VIEW/{par}/{view}/{lat}/")
        # if os.path.exists(f"D:/OneDrive - Industrial University of HoChiMinh City/WORKBASE/Project-rsna-breast-cancer-detection/Data_main/Dataset_VIEW/{par}/{lat}/{view}/") == False:
        #     os.mkdir(f"D:/OneDrive - Industrial University of HoChiMinh City/WORKBASE/Project-rsna-breast-cancer-detection/Data_main/Dataset_VIEW/{par}/{lat}/{view}/")
        # try:
        # img = cv2.resize(img, (512, 512))
        # try:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # except:
        #     img = img
        print(1)
        imageio.imwrite(f"D:\OneDrive - Industrial University of HoChiMinh City\WORKBASE\Project-rsna-breast-cancer-detection\Data_main\CSV_MAIN\Rate11\{self.stores}/{self.meta.patient_id[id]}_{self.meta.image_id[id]}.png", img)

    def process_move(self, id):
        self.xx(id)
    def my_method(self, ls):
        with mp.Pool(12) as p:
            p.map(self.process_move, ls)
    