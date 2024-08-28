from __future__ import print_function

import argparse
import os, cv2, sys
import time

from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras import optimizers
import numpy as np
import glob
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

start_time = time.time()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input directory name', nargs='?', default="input/")
parser.add_argument('-o', '--output', type=str, help='output file name', nargs='?', default="result")


IMAGE_FILE_PATH_DISTORTED = parser.parse_args().input
path_to_weights = 'weights/Regression/Single_net/weights_10_0.02.h5'
IMAGE_SIZE = 299
INPUT_SIZE = 299


focal_end = 400
focal_start = 40
if parser.parse_args().output == "result":
    filename_results = f'results_{focal_start}to{focal_end}.txt'
else:
    filename_results = f'{parser.parse_args().output}.txt'

if os.path.exists(filename_results):
    sys.exit("file exists already")

classes_focal = list(np.arange(focal_start, focal_end + 1, 10))
classes_distortion = list(np.arange(0, 61, 1) / 50.)


def get_paths(image_file_path_distorted):
    print(f"Looking for images in {image_file_path_distorted}")
    _paths_test = glob.glob(image_file_path_distorted + "*.JPG")
    print(f"Found {len(_paths_test)} images")
    _paths_test.sort()

    return _paths_test


paths_test = get_paths(IMAGE_FILE_PATH_DISTORTED)

print(len(paths_test), 'test samples')

with tf.device('/gpu:0'):
    input_shape = (299, 299, 3)
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=main_input, input_shape=input_shape)
    phi_features = phi_model.output
    phi_flattened = Flatten(name='phi-flattened')(phi_features)
    final_output_focal = Dense(1, activation='sigmoid', name='output_focal')(phi_flattened)
    final_output_distortion = Dense(1, activation='sigmoid', name='output_distortion')(phi_flattened)

    layer_index = 0
    for layer in phi_model.layers:
        layer.name = layer.name + "_phi"

    model = Model(input=main_input, output=[final_output_focal, final_output_distortion])
    model.load_weights(path_to_weights)

    n_acc_focal = 0
    n_acc_dist = 0
    print(len(paths_test))
    file = open(filename_results, 'a')
    for i, path in enumerate(paths_test):
        if i % 1000 == 0:
            print(i, ' ', len(paths_test))
        image = cv2.imread(path)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        image = np.expand_dims(image, 0)

        image = preprocess_input(image)

        # loop
        prediction_focal = model.predict(image)[0]
        prediction_dist = model.predict(image)[1]

        curr_focal_pred = (prediction_focal[0][0] * (focal_end + 1. - focal_start * 1.) + focal_start * 1.) * (
                IMAGE_SIZE * 1.0) / (INPUT_SIZE * 1.0)
        curr_dist_pred = prediction_dist[0][0] * 1.2
        file.write(f"{path}\t{str(curr_focal_pred)}\t{str(curr_dist_pred)}\n")

    print('focal:')
    print(n_acc_focal)
    print(len(paths_test))
    print(n_acc_focal * 1.0 / (len(paths_test) * 1.0))

    print('dist:')
    print(n_acc_dist)
    print(len(paths_test))
    print(n_acc_dist * 1.0 / (len(paths_test) * 1.0))
    file.close()

end_time = time.time()
print(f"Time spent : {end_time - start_time}")