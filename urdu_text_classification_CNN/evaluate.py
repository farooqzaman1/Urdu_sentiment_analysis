import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# load datasets
trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')

# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
print(trainX.shape, testX.shape)

# load the model
model = load_model('model.h5')

# evaluate model on training dataset
loss, acc = model.evaluate([trainX, trainX, trainX], array(trainLabels), verbose=0)
print('Train Accuracy: %f' % (acc * 100))

# evaluate model on test dataset dataset
loss, acc = model.evaluate([testX, testX, testX], array(testLabels), verbose=0)
print('Test Accuracy: %f' % (acc * 100))

y_pred = model.predict([testX, testX, testX], verbose=0)


reverse_mapping = {0: "neu", 1: "neg", 2: "pos"}
y_pred = [reverse_mapping[np.argmax(y) ]for y in y_pred]

y_test = [reverse_mapping[np.argmax(y)] for y in testLabels]

print(y_pred[:20])
