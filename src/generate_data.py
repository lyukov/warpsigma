################################################################################
# Generate dataset.
# This file creates:
# 1. *.npy file with (N, w, h)-shaped numpy array 
# 2. *.csv file with 'name' - number of row in .npy file,
#    'label' - sigma, the "right answer" for this sample
#    'file' - reference filename
#
# Dmitry Lyukov, 2018
################################################################################

from extract_patches import create_patch_coords_generator
from extract_patches import create_patch_coords_generator_from_mse_dispersion
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import progressbar as pbar

################################################################################

available_sigmas = np.arange(0.5, 6.5, 0.5)

def create_filename_generator(ref_dir, distorted_dir, restored_dir):
    for ref_name in os.listdir(ref_dir):
        ref_path = os.path.join(ref_dir, ref_name)
        number = ref_name.split('.')[0][1:]

        dist_names = list(filter(
            lambda x: ('src_i' + number) in x,
            os.listdir(distorted_dir) ))

        for dist_name in dist_names:
            dist_path = os.path.join(distorted_dir, dist_name)

            restored_paths = list(map(
                lambda x: os.path.join(restored_dir, 'warp_' + str(x), dist_name),
                available_sigmas ))

            yield ref_path, dist_path, restored_paths, ref_name

def create_image_generator(ref_dir, distorted_dir, restored_dir):
    fname_gen = create_filename_generator(ref_dir, distorted_dir, restored_dir)
    for ref_path, dist_path, restored_paths, ref_name in fname_gen:
        ref_img = rgb2gray(imread(ref_path))
        dist_img = rgb2gray(imread(dist_path))
        restored_imgs = list(map(lambda x: rgb2gray(imread(x)), restored_paths))
        yield ref_img, dist_img, restored_imgs, ref_name

def create_patch_generator(ref_img, dist_img, restored_imgs, W, H):
        #patch_coords_gen = create_patch_coords_generator(ref_img, W, H, 0.05, 3, sigma)

        patch_coords_gen = create_patch_coords_generator_from_mse_dispersion(
            ref_img, restored_imgs, W, H, ratio=0.05, step=3)

        for top, left in patch_coords_gen:
            ref_patch    = ref_img [top : top + H, left : left + W]
            dist_patch   = dist_img[top : top + H, left : left + W]
            rest_patches = list(map(
                lambda x: x[top : top + H, left : left + W],
                restored_imgs ))
            yield ref_patch, dist_patch, rest_patches

################################################################################

def mse(a, b):
    return np.sqrt(((a - b) ** 2).sum())

# returns sigma of nearest restored
def get_label(ref, restored_list):
    num = np.argmin(list(map(lambda x: mse(ref, x), restored_list)))
    return (num + 1) * 0.5

################################################################################

def generate_dataset(ref_dir, distorted_dir, restored_dir,
                     dataset_fname, labels_fname, W, H, sigma=3.0):
    names = []
    labels = []
    ref_names = []
    data = []

    image_gen = create_image_generator(ref_dir, distorted_dir, restored_dir)
    image_tuples = list(image_gen)

    t = 0
    progress_bar = pbar.ProgressBar(len(image_tuples))
    progress_bar.start()
    sample_num = 0

    for ref_img, dist_img, restored_imgs, ref_name in image_tuples:

        patch_gen = create_patch_generator(ref_img, dist_img, restored_imgs, W, H)
        
        for ref, dist, restored in patch_gen:
            label = get_label(ref, restored)
            data.append(dist)
            names.append(sample_num)
            labels.append(label)
            ref_names.append(ref_name)
            sample_num += 1
            
        t += 1
        progress_bar.update(t)

    dataframe = pd.DataFrame(
        {'name' : names, 'label' : labels, 'file': ref_names}).set_index('name')
    dataframe.to_csv(labels_fname)

    data = np.array(data)
    data = (data * (2 ** 16 - 1)).astype('uint16')
    data.tofile(dataset_fname)

################################################################################

root_dir      = '.'
data_dir      = os.path.join(root_dir, 'data')
ref_dir       = os.path.join(data_dir, 'reference')
distorted_dir = os.path.join(data_dir, 'distorted')
restored_dir  = os.path.join(data_dir, 'restored')
dataset_dir   = os.path.join(data_dir, 'dataset')

dataset_fname = os.path.join(dataset_dir, 'dataset_mse.npy')
labels_fname  = os.path.join(dataset_dir, 'labels_mse.csv')

generate_dataset(ref_dir, distorted_dir, restored_dir,
                 dataset_fname, labels_fname, 31, 31)

################################################################################