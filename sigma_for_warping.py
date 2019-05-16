from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from skimage.color import rgb2gray
from skimage.io import imread, imsave, imshow
import keras
import matplotlib.pyplot as plt
import numpy as np

################################################################################

# Model loading
model = Sequential()
model.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(None, None, 1)))
model.add(Convolution2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(64, (11, 11), activation='relu'))
model.add(Convolution2D(17, (1, 1), activation='relu'))
model.add(Convolution2D(1, (1, 1), activation='relu'))
adam = Adam(lr=0.0003)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_absolute_error'])
model.load_weights('model12.h5')

################################################################################

# Model predicts values in logarithmic scale.
# Here we transform values to each other.

def k_from_sigma(s):
    return 4.0 * np.log2(s) + 4

def sigma_from_k(k):
    return 2 ** (k / 4 - 1)

################################################################################

def predict_for_full_image(img, model):
    k_map = model.predict(img.reshape(1, *img.shape, 1))
    k_map = k_map.reshape(k_map.shape[1:-1])
    sigma_map = sigma_from_k(k_map)
    return sigma_map

################################################################################

def get_sigma_map(img, model):
    sigma_map = np.zeros(img.shape)
    of = 15 # offset

    tmp_sigma_map = predict_for_full_image(img[0:, 0:], model)
    tmp_shape = tmp_sigma_map.shape
    sigma_map[of: of + tmp_shape[0] * 2: 2,
              of: of + tmp_shape[1] * 2: 2] = tmp_sigma_map

    tmp_sigma_map = predict_for_full_image(img[0:, 1:], model)
    tmp_shape = tmp_sigma_map.shape
    sigma_map[of: of + tmp_shape[0] * 2: 2,
              of + 1: of + 1 + tmp_shape[1] * 2: 2] = tmp_sigma_map

    tmp_sigma_map = predict_for_full_image(img[1:, 0:], model)
    tmp_shape = tmp_sigma_map.shape
    sigma_map[of + 1: of + 1 + tmp_shape[0] * 2: 2,
              of: of + tmp_shape[1] * 2: 2] = tmp_sigma_map

    tmp_sigma_map = predict_for_full_image(img[1:, 1:], model)
    tmp_shape = tmp_sigma_map.shape
    sigma_map[of + 1: of + 1 + tmp_shape[0] * 2: 2,
              of + 1: of + 1 + tmp_shape[1] * 2: 2] = tmp_sigma_map

    sigma_map = np.pad(sigma_map[of: -of, of: -of], of, mode='edge')

    return sigma_map

################################################################################

# Example

img = imread('level1_src_i23gauss.png')
img = rgb2gray(img)

sigma_map = get_sigma_map(img, model)

# Save as binary file.
sigma_map.tofile('sigma_map.npy')

# Or save as png with clipping at 6.0 and transforming [0, 6] -> [0, 255]
imsave('sigma_map.png', np.clip(sigma_map / 6.0, 0, 1))