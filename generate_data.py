import numpy as np
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.color import rgb2gray
from extract_images import create_patch_coords_generator
from extract_images import create_patch_coords_generator_from_mse_dispersion

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

def create_patch_generator(ref_dir, distorted_dir, restored_dir, W, H, sigma=3.0):
    image_gen = create_image_generator(ref_dir, distorted_dir, restored_dir)
    for ref_img, dist_img, restored_imgs in image_gen:
        #patch_coords_gen = create_patch_coords_generator(ref_img, W, H, 0.05, 3, sigma)

        patch_coords_gen = create_patch_coords_generator_from_mse_dispersion(
            ref_img, restored_imgs, W, H, 0.05, 3)

        for top, left in patch_coords_gen:
            ref_patch  = ref_img [top : top + H, left : left + W]
            dist_patch = dist_img[top : top + H, left : left + W]
            rest_patch = list(map(lambda x: x[top : top + H, left : left + W], restored_imgs))
            yield ref_patch, dist_patch, rest_patch

from math import sqrt
def mse(a, b):
    return sqrt(((a - b) ** 2).sum())

def get_label(ref, restored_list):
    num = np.argmin(list(map(lambda x: mse(ref, x), restored_list)))
    return (num + 1) * 0.5

def generate_dataset(ref_dir, distorted_dir, restored_dir,
                     dataset_fname, labels_fname, W, H, sigma=3.0):
    names = []
    labels = []
    data = []
    patch_gen = create_patch_generator(ref_dir, distorted_dir, restored_dir, W, H, sigma)
    i = 0
    for ref, dist, restored in patch_gen:
        label = get_label(ref, restored)
        data.append(dist)
        names.append(i)
        labels.append(label)
        i += 1
        if (i % 10000 == 0):
            print(i, 'patches have been created')
    pd.DataFrame({'name' : names, 'label' : labels}).set_index('name').to_csv(labels_fname)
    data = np.array(data)
    data = (data * (2 ** 16 - 1)).astype('uint16')
    np.array(data).tofile(dataset_fname)

data_dir      = 'data'
ref_dir       = os.path.join(data_dir, 'reference')
distorted_dir = os.path.join(data_dir, 'distorted')
restored_dir  = os.path.join(data_dir, 'restored')
dataset_dir   = os.path.join(data_dir, 'dataset')

dataset_fname = os.path.join(dataset_dir, 'dataset_mse.npy')
labels_fname  = os.path.join(dataset_dir, 'labels_mse.csv')

generate_dataset(ref_dir, distorted_dir, restored_dir,
                 dataset_fname, labels_fname, 31, 31, sigma=3.0)