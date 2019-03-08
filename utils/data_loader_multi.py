from keras.layers import Input
from keras.layers.merge import add
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import skimage.measure as measure

from astropy.io import fits
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os, sys, csv, random, pickle
from datetime import datetime

from pickle_helper import pickle_load

CLASS_NUM = 7
IMG_CHANNELS = 1
IMG_SIZE = 50

INPUT_SIZE = (50,50, IMG_CHANNELS)

#Hyper parameters
LEARNING_RATE = 0.0001
EPOCHS = 60
BATCH_SIZE = 128

FILE_HOME = "/Users/sheep/Documents/research/project/hsc"
DATA_SOURCE = 'lae_fits_reduced50000_png.pickle'
TEST_DATA_SOURCE = 'real_observed_png.pickle'
#DATA_SOURCE = 'lae_fits.pickle'

IMG_IDX = 2
LABEL_IDX = 3


class DatasetLoader:

    def __init__(self, csv_file_path, root_dir):
        data_frame = pd.read_csv(csv_file_path, header=None, dtype={'id':'object', 'path':'object', 'label':'int', 'bool_label':'int'})
        self.root_dir = root_dir
        self.dataset = [ None for i in range(CLASS_NUM) ]
        for i in range(CLASS_NUM):
            tmp_data_frame = data_frame[data_frame[LABEL_IDX]==i]
            self.dataset[i] = self.create_dataset(tmp_data_frame)

    def create_dataset(self, data_frame):
        data_list = []

        for idx, row_data in data_frame.iterrows():
            img_no = str(row_data[0])
            img_path1 = row_data[1][:-4] + 'png'
            img_path2= row_data[2][:-4] + 'png'
            label = row_data[3]
            bool_label = row_data[4]
            try:
                image1 = self.load_png_image(self.root_dir + img_path1)
                image2 = self.load_png_image(self.root_dir + img_path2)
                if image1.shape != (77,77) or image2.shape != (77,77):
                    print(image1.shape)
                    print(img_path1)
                    print(image2.shape)
                    print(img_path2)
                else:
                    image1 = self.crop_center(image1, 50, 50)
                    image2 = self.crop_center(image2, 50, 50)
                #image = measure.block_reduce(image, (20,20), np.median) 
                #image = self.normalize2(image)
                #image_feature = self.extract_feature(( normalized_image + 0.5 )*10)
                #print(image_feature, label)

                #saveimg = Image.fromarray(np.uint8(image))
                #saveimg.save(self.root_dir + img_path)

                #print(idx)
                    data_list.append( (image1, image2, label, bool_label, img_no, img_path1, img_path2) )
            except FileNotFoundError:
                print('fn ' + self.root_dir + img_path1)
                print('fn ' + self.root_dir + img_path2)
            except IOError:
                print('io ' + self.root_dir + img_path1)
                print('io ' + self.root_dir + img_path2)

        return data_list

    def crop_center(self, img,cropx, cropy):
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[starty:starty+cropy,startx:startx+cropx]

    def get_dataset(self, label):
        return self.dataset[label]

    def load_image(self, img_path):
        row_image = fits.getdata(img_path)
        image = np.copy(row_image)
        return image

    def load_png_image(self, img_path):
        with Image.open(img_path) as row_image:
            image = np.asarray(row_image)
            if image.shape != (77,77,4):
                print(image.shape)
                print(img_path)
            return image[...,1]

    def get_train_test_data(self, label, num, shuffle=False):
        data_list = self.get_dataset(label)
        idx_list = [idx for idx in range(len(data_list))]
        if shuffle:
            random.shuffle(idx_list)
        train_idx = idx_list[:num]
        test_idx = idx_list[num:]
        train_data_list = [data_list[idx] for idx in train_idx]
        test_data_list = [data_list[idx] for idx in test_idx]
        return train_data_list, test_data_list