################################################################################

from math import floor
from progressbar import ProgressBar
from skimage.color import rgb2gray
from skimage.io import imread, imsave, imshow
import keras
import matplotlib.pyplot as plt
import numpy as np
import os

################################################################################

available_sigmas = np.arange(0.5, 6.5, 0.5)

def k_from_sigma(s):
    return 4.0 * np.log2(s) + 4

def sigma_from_k(k):
    return 2 ** (k / 4 - 1)

def mse(a, b):
    return np.sqrt(((a - b) ** 2).sum())

################################################################################

# Predict for each pixel
def get_k_map(img, model, H, W):
    res_shape = (img.shape[0] - H + 1, img.shape[1] - W + 1)
    res = np.zeros(res_shape)

    progress_bar = ProgressBar(res_shape[0])
    progress_bar.start()

    for i in range(res_shape[0]):
        for j in range(res_shape[1]):
            patch = img[i : i + 31, j : j + 31]
            patch = patch.reshape((1, 31, 31, 1))
            res[i, j] = model.predict(patch)[0]
        progress_bar.update(i + 1)
    return res

################################################################################

def restore_ij(i, j, sigma, restored_list):
    image_n = (sigma - 0.5) * 2
    floor_n = floor(image_n)
    if (floor_n >= len(restored_list) - 1):
        floor_n = len(restored_list) - 2
    delta_n = sigma - floor_n
    # Linear interpolation
    return (restored_list[floor_n + 1][i, j] * delta_n +
           restored_list[floor_n][i, j] * (1 - delta_n))

def restore(img, sigma_map, restored_list):
    new_sigma_map = np.zeros(img.shape)
    new_sigma_map[0: 15, 0: 15] *= sigma_map[0 ,  0]
    new_sigma_map[-15: , 0: 15] *= sigma_map[-1,  0]
    new_sigma_map[0: 15, -15: ] *= sigma_map[0 , -1]
    new_sigma_map[-15: , -15: ] *= sigma_map[-1, -1]
    new_sigma_map[15: -15, 15: -15] = sigma_map

    result = np.zeros(img.shape)
    progress_bar = ProgressBar(img.shape[0])
    progress_bar.start()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = restore_ij(i, j, new_sigma_map[i, j], restored_list)
        progress_bar.update(i + 1)
    return result

################################################################################

root_dir      = '.'
data_dir      = os.path.join(root_dir, 'data')
ref_dir       = os.path.join(data_dir, 'reference')
distorted_dir = os.path.join(data_dir, 'distorted')
restored_dir  = os.path.join(data_dir, 'restored')
model_dir     = os.path.join(root_dir, 'models')

################################################################################

def load_model(name):
    json_file = open(os.path.join(model_dir, name + ".json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(model_dir, name + ".h5"))
    return loaded_model

################################################################################

# Model loading
model = load_model('model8')

# Image loading
s_img_num = '21'

ref_path  = os.path.join(ref_dir, 'I' + s_img_num + '.BMP')
reference = rgb2gray(imread(ref_path))

dist_path = os.path.join(distorted_dir, 'level1_src_i' + s_img_num + 'ring.png')
distorted = rgb2gray(imread(dist_path))

restored_list = list(map(
    lambda x:
        rgb2gray(imread(restored_dir + '/warp_' + str(x) +
                   '/level1_src_i' + s_img_num + 'ring.png')),
    available_sigmas
    ))

# Restore image
k_map = get_k_map(distorted, model, 31, 31)
sigma_map = sigma_from_k(k_map)

restored = restore(distorted, sigma_map, restored_list)

################################################################################