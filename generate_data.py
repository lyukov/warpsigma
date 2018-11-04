import numpy as np
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.color import rgb2gray

def create_filename_generator(ref_dir, distorted_dir, restored_dir):
    for ref_name in os.listdir(ref_dir):
        ref_path = os.path.join(ref_dir, ref_name)
        number = ref_name.split('.')[0][1:]
        dist_names = list(filter(lambda x: ('src_i' + number) in x, os.listdir(distorted_dir)))
        for dist_name in dist_names:
            dist_path = os.path.join(distorted_dir, dist_name)
            restored_paths = list(map(lambda x: os.path.join(restored_dir, 'warp_' + str(x), dist_name), np.arange(0.5, 6.5, 0.5)))
            yield ref_path, dist_path, restored_paths

def create_image_generator(ref_dir, distorted_dir, restored_dir):
    fname_gen = create_filename_generator(ref_dir, distorted_dir, restored_dir)
    for ref_path, dist_path, restored_paths in fname_gen:
        ref_img = rgb2gray(imread(ref_path))
        dist_img = rgb2gray(imread(dist_path))
        restored_imgs = list(map(lambda x: rgb2gray(imread(x)), restored_paths))
        yield ref_img, dist_img, restored_imgs

def create_batch_generator(ref_dir, distorted_dir, restored_dir, W, H):
    image_gen = create_image_generator(ref_dir, distorted_dir, restored_dir)
    for ref_img, dist_img, restored_imgs in image_gen:
        for k in range(ref_img.shape[0] // H):
            for l in range(ref_img.shape[1] // W):
                ref_batch = ref_img[k * H : (k+1) * H, l * W : (l+1) * W]
                dist_batch = dist_img[k * H : (k+1) * H, l * W : (l+1) * W]
                rest_batch = list(map(lambda x: x[k * H : (k+1) * H, l * W : (l+1) * W], restored_imgs))
                yield ref_batch, dist_batch, rest_batch

from math import sqrt
def mse(a, b):
    return sqrt(((a - b) ** 2).sum())

def get_label(ref, restored_list):
    num = np.argmin(list(map(lambda x: mse(ref, x), restored_list)))
    return (num + 1) * 0.5

def generate_dataset(ref_dir, distorted_dir, restored_dir, dataset_dir, W, H):
    labels_path = os.path.join(dataset_dir, 'labels.csv')
    filenames = []
    labels = []
    batch_gen = create_batch_generator(ref_dir, distorted_dir, restored_dir, W, H)
    i = 0
    for ref, dist, restored in batch_gen:
        label = get_label(ref, restored)
        filename = str(i) + '.png'
        imsave(os.path.join(dataset_dir, filename), dist)
        filenames.append(filename)
        labels.append(label)
        i += 1
        if (i % 100 == 0):
            pd.DataFrame({'filename' : filenames, 'label' : labels}).set_index('filename').to_csv(labels_path)
    pd.DataFrame({'filename' : filenames, 'label' : labels}).set_index('filename').to_csv(labels_path)

data_dir      = 'data'
ref_dir       = os.path.join(data_dir, 'reference')
distorted_dir = os.path.join(data_dir, 'distorted')
restored_dir  = os.path.join(data_dir, 'restored')
dataset_dir   = os.path.join(data_dir, 'dataset')

generate_dataset(ref_dir, distorted_dir, restored_dir, dataset_dir, 31, 31)