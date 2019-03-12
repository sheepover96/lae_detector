from astropy.io import fits
from astropy.visualization import ZScaleInterval
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os, sys, csv, random, pickle
from datetime import datetime

norm = ZScaleInterval()

IMG_SIZE = 50
ID_IDX = 0
IMG_IDX1 = 1
IMG_IDX2 = 2


class MultiBandDataLoader:

    def __init__(self, csv_file_path):
        data_frame = pd.read_csv(csv_file_path, header=None, dtype={'id':'object', 'path1':'object', 'path2':'object'})
        self.dataset = self.create_dataset(data_frame)

    def create_dataset(self, data_frame):
        data_list = []

        for idx, row_data in data_frame.iterrows():
            img_no = str(row_data[ID_IDX])
            img_path1 = row_data[IMG_IDX1]
            img_path2= row_data[IMG_IDX2]
            try:
                image1 = self.load_image(img_path1)
                image1 = self.crop_center(image1, IMG_SIZE, IMG_SIZE)
                image1 = norm(image1)

                image2 = self.load_image(img_path2)
                image2 = self.crop_center(image2, IMG_SIZE, IMG_SIZE)
                image2 = norm(image2)

                data_list.append( (image1, image2, img_no, img_path1, img_path2) )
            except FileNotFoundError:
                print('FineNotFound: ', img_path1)
                print('FineNotFound: ', img_path2)
            except IOError:
                print('IOError: ', img_path1)
                print('IOError: ', img_path2)

        return data_list

    def crop_center(self, img,cropx, cropy):
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[starty:starty+cropy,startx:startx+cropx]

    def get_dataset(self):
        return self.dataset

    def load_image(self, img_path):
        raw_image = fits.getdata(img_path)
        return raw_image
