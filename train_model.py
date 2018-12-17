################################################################################

import numpy as np
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.color import rgb2gray
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import CSVLogger

np.random.seed(42)

################################################################################

def k_from_sigma(s):
    return 4.0 * np.log2(s) + 4

def sigma_from_k(k):
    return 2 ** (k / 4 - 1)

def get_smoothed_y(mu, sigma, n_classes):
    return np.exp(-((np.array(range(n_classes)) - mu) ** 2) / (2 * sigma ** 2))

################################################################################

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, dataset_fname, batch_size=32, dim=(31,31),
                 n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.dataset_fname = dataset_fname
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        data = np.fromfile(dataset_fname, dtype='uint16')
        data = data.astype(float) / (2 ** 16 - 1)
        N = len(data) // np.prod(dim)
        self.data = data.reshape((N, *dim))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __load_image(self, ID):
        return self.data[int(ID)]
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            sample = self.__load_image(ID)
            sample = sample.reshape((*self.dim, self.n_channels))
            X[i,] = sample

            # Store class
            sigma_gt = self.labels[ID]
            k_gt = k_from_sigma(sigma_gt)
            y[i] = k_gt

        return X, y

################################################################################

def save_model(model, name):
    model_json = model.to_json()
    with open(os.path.join(model_dir, name + ".json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(model_dir, name + ".h5"))

def load_model(name):
    json_file = open(os.path.join(model_dir, name + ".json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(model_dir, name + ".h5"))

################################################################################

root_dir    = '.'
data_dir    = os.path.join(root_dir, 'data')
model_dir   = os.path.join(root_dir, 'models')
dataset_dir = os.path.join(data_dir, 'dataset')

# Data loading
labels_fname  = 'labels_mse.csv'
dataset_fname = 'dataset_mse.npy'

labels_path = os.path.join(dataset_dir, labels_fname)
data = pd.read_csv(labels_path).set_index('name')

# Shuffle data before dividing to train and validation
#data = data.sample(frac=1)

# Shuffle labels
#data['label'] = data['label'].sample(frac=1).reset_index()['label']
#data.hist()

# Dividing to train and validation
N = len(data)
validation = list(data.index[:N // 6])
train = list(data.index[N // 6:])
print("Validation", len(validation), ", Train", len(train), "N ", N)

# Parameters
params = {'dim': (31, 31),
          'batch_size': 64,
          'dataset_fname' : os.path.join(dataset_dir, dataset_fname),
          'n_channels': 1,
          'shuffle': True}

# Dataset
labels = data.to_dict()['label']

# Generators
training_generator = DataGenerator(train, labels, **params)
validation_generator = DataGenerator(validation, labels, **params)

################################################################################

# Design model
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(31, 31, 1),
    kernel_initializer='normal'))
model.add(Convolution2D(32, (3, 3), activation='relu',
    kernel_initializer='normal'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Convolution2D(128, (13, 13), activation='relu',
    kernel_initializer='normal'))
model.add(Dropout(0.1))
model.add(Convolution2D(17, (1, 1), activation='relu',
    kernel_initializer='normal'))
model.add(Dropout(0.4))
model.add(Convolution2D(1, (1, 1), activation='relu',
    kernel_initializer='normal'))
model.add(Flatten())

print('Output shape:', model.output_shape)
print('Parameters:', model.count_params())

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_absolute_error'])

################################################################################

model_name = 'regr1'
log_path = os.path.join(root_dir, model_name + '.csv')

# Train model on dataset
history = model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=4,
    nb_epoch=1,
    callbacks=[CSVLogger(log_path)]
)

save_model(model, model_name)

history = pd.DataFrame(history.history)
history[['mean_absolute_error', 'val_mean_absolute_error']].plot(grid=True)
#plt.show()
plt.savefig(os.path.join(root_dir, model_name + '.pdf'))

################################################################################