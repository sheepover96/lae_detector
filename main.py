from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.preprocessing import image

import pandas as pd
import numpy as np
import cv2
import csv, argparse

from utils.data_loader_multi import MultiBandDataLoader

MODEL_PATH = 'model/cnn_model.json'
WEIGHTS_PATH = 'model/cnn_weights.hdf5'

parser = argparse.ArgumentParser(description='Subaru HSC: LAE detector')
parser.add_argument('data_path', help='path to csv dataset')

args = parser.parse_args()
DATASET_PATH = args.data_path

def predict(model, test_data):
    test_data_np = np.array(test_data)
    test_data_np = test_data_np.reshape(test_data_np.shape[0], 50, 50, 1)
    #score = self.model.evaluate(test_data_np, test_label_np, verbose=0)
    prob = model.predict(test_data_np)
    pred = model.predict_classes(test_data_np)
    return None, pred, prob

def write_multiband_result(dataset, pred_narrow_result_list, pred_g_result_list, pred_multiband_result_list, prob_narrow_list, prob_g_list, output_path):
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for data_row, pred_narrow_result, pred_g_result, pred_multiband_result, prob_narrow, prob_g in zip(dataset, pred_narrow_result_list, pred_g_result_list, pred_multiband_result_list, prob_narrow_list, prob_g_list):
            label = data_row[2]
            image_number = data_row[4]
            image_path1 = data_row[5]
            image_path2 = data_row[6]
            row = [image_number, image_path1, image_path2, label, pred_narrow_result, pred_g_result, pred_multiband_result, prob_narrow, prob_g]
            writer.writerow(row)


optimizer = Adam(lr=0.001)
loss = 'categorical_crossentropy'
cnn_metrics = ['accuracy']
with open(MODEL_PATH, 'r') as f_model:
    json_model = f_model.read()
cnn_model = model_from_json(json_model)
cnn_model.compile(loss=loss, optimizer=optimizer, metrics=cnn_metrics)
cnn_model.load_weights(WEIGHTS_PATH)

data_loader = MultiBandDataLoader(DATASET_PATH)
dataset = data_loader.get_dataset()

img_narrow = [data[0] for data in dataset]
img_g = [data[1] for data in dataset]

_, pred_narrow, prob_narrow = predict(cnn_model, img_narrow)
_, pred_g, prob_g = predict(cnn_model, img_g)

pred_multiband = []
for narrow_pred, g_perd in zip(pred_narrow, pred_g):
    pred = 1
    if not( narrow_pred == 1 and g_perd==0):
        pred = 0
    pred_multiband.append(pred)

write_multiband_result(dataset, pred_narrow, pred_g, pred_multiband, prob_narrow, prob_g, './result/multiband_test_classw_400_17.csv')
