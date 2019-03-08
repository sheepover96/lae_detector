from keras import layers
from keras.layers import Input
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
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
        data_frame = pd.read_csv(csv_file_path, header=None)
        self.root_dir = root_dir
        self.dataset = [ None for i in range(CLASS_NUM) ]
        for i in range(CLASS_NUM):
            tmp_data_frame = data_frame[data_frame[LABEL_IDX]==i]
            self.dataset[i] = self.create_dataset(tmp_data_frame)

    def create_dataset(self, data_frame):
        data_list = []

        for idx, row_data in data_frame.iterrows():
            img_no = str(row_data[0])
            img_path = row_data[1]
            label = row_data[2]
            bool_label = row_data[3]
            try:
                image = self.load_image(img_path)
                image = self.crop_center(image, IMG_SIZE, IMG_SIZE)
                #image = self.normalize2(image)
                #image_feature = self.extract_feature(( normalized_image + 0.5 )*10)
                #print(image_feature, label)

                data_list.append( (image, label, bool_label, img_no, img_path) )
            except FileNotFoundError:
                print(self.root_dir + img_path)

        return data_list

    def crop_center(self, img, cropx, cropy):
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[starty:starty+cropy,startx:startx+cropx]

    def get_dataset(self, label):
        return self.dataset[label]

    def load_image(self, img_path):
        row_image = fits.getdata(self.root_dir + img_path)
        image = np.copy(row_image)
        return image

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
