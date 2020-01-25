import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.utils import np_utils

from pymongo import MongoClient
client = MongoClient()
db = client['audio']
tracks = db.tracks

def open_1d_spectrograms(gender):
    """ Tracks are sometimes missing frequency data.  After flattening,
        shorten the tracks and make sure they aren't too small.
        Flatten the spectrogram to 1d.
    """
    flatlen = 307260
    mel_tracks = tracks.find({'component':'vocals.wav', 'gender':gender})
    out = np.zeros((mel_tracks.count(), flatlen))
    for i, path in enumerate(map(lambda track: track['mel_path'], mel_tracks)):
        loaded = np.loadtxt(path).flatten()[:307260]
        if loaded.shape[0]  == 307260:
            out[i] = loaded
    return out

male_mel = open_1d_spectrograms('male')
female_mel = open_1d_spectrograms('female')

X = np.concatenate([female_mel, male_mel])
X = X.reshape(*X.shape, -1)
y = [1,]*female_mel.shape[0] + [0,]*male_mel.shape[0]

X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size = .20, random_state = 0)
)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.20, random_state = 0)
y_train_cat = np_utils.to_categorical(y_train)
y_val_cat = np_utils.to_categorical(y_val)


def conv1d(X_train, y_train_cat, validation_data=None):

    NN = Sequential()

    # Conv block 1
    NN.add(Conv1D(filters = 32, kernel_size = 9, activation='relu', 
                  input_shape = X_train.shape[1:]))
    NN.add(Conv1D(filters = 32, kernel_size = 9, activation='relu'))
    NN.add(MaxPooling1D(pool_size=4))

    # Conv block 2 - note we increase filter dimension as we move
    # further into the network
    NN.add(Conv1D(filters = 48, kernel_size = 9, activation='relu'))
    NN.add(Conv1D(filters = 48, kernel_size = 9, activation='relu'))
    NN.add(MaxPooling1D(pool_size=4))

    NN.add(Conv1D(filters = 120, kernel_size = 9, activation='relu'))
    NN.add(Conv1D(filters = 120, kernel_size = 9, activation='relu'))
    NN.add(MaxPooling1D(pool_size=4))

    # Fully connected block - flattening followed by dense and output layers.
    NN.add(Flatten())
    NN.add(Dense(128, activation='relu'))
    NN.add(Dense(64, activation='relu'))
    NN.add(Dense(2, activation='sigmoid'))

    NN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    NN.summary()
    NN.fit(
        X_train, y_train_cat, epochs=35, batch_size=256, verbose=1,
        validation_data=validation_data
    )
    return NN
    
model_cnn = conv1d(X_train, y_train_cat, validation_data=(X_val, y_val_cat))
model_cnn.save('cnn.model')