from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from sklearn import preprocessing
import numpy as np

from keras.utils import to_categorical

from pickle import dump
import pandas  as pd
from sklearn.model_selection import train_test_split


def read_data():
    data = pd.read_excel("data/data_binary.xlsx")
    from sklearn.utils import shuffle
    data = shuffle(data)
    X = data['Text'].values
    y = data['Label'].values
    return X, y


# save a dataset to file
def save_dataset(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)


def Label_encoder_catagorical(y, mapping):
    y = np.array([mapping[i] for i in y])
    clases = len(mapping)
    y = to_categorical(y, num_classes=clases, dtype='float32')
    return y


X, y = read_data()

mapp = {"neg": 0, "pos": 1}

y = Label_encoder_catagorical(y, mapping=mapp)

trainX, testX, trainy, testY = train_test_split(X, y, test_size=0.33, random_state=42)

save_dataset([trainX, trainy], 'train.pkl')
save_dataset([testX, testY], 'test.pkl')
