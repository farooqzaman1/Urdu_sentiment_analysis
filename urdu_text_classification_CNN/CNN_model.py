import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.utils.vis_utils import plot_model

from tensorflow import keras
from tensorflow.keras.utils import plot_model
import tensorflow as  tf

from tensorflow.keras import layers
#####################################
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from matplotlib import pyplot as plt
import seaborn as sns
########################################
def evaluate_model(y_test, y_pred, labels, title='Confusion Matrix', name="123", width=5, height=5):
    f1_macro = f1_score(y_test, y_pred, average='macro') * 100
    f1_micro = f1_score(y_test, y_pred, average='micro') * 100
    f1_weighted = f1_score(y_test, y_pred, average='weighted') * 100
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_macro = np.round(f1_macro, 2)
    f1_micro = np.round(f1_micro, 2)
    f1_weighted = np.round(f1_weighted, 2)
    accuracy = np.round(accuracy, 2)
    macro_p = precision_score(y_test, y_pred, average='macro') * 100
    micro_p = precision_score(y_test, y_pred, average='micro') * 100
    macro_r = recall_score(y_test, y_pred, average='macro') * 100
    micro_r = recall_score(y_test, y_pred, average='micro') * 100
    macro_p = np.round(macro_p, 2)
    micro_p = np.round(micro_p, 2)
    macro_r = np.round(macro_r, 2)
    micro_r = np.round(micro_r, 2)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig = plt.figure(figsize=(width, height))
    ax = plt.subplot()
    ax = sns.heatmap(cm, annot=True, ax=ax, fmt='d');  # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels\n'
                  + 'F1=[macro:' + str(f1_macro) + "  micro:" + str(f1_micro) + "]   " \
                  + "P = [macro:" + str(macro_p) + "  micro:" + str(micro_p) + "]\n" \
                  + "R = [macro:" + str(macro_r) + "  micro:" + str(micro_r) + "]                   " \
                  + "Accuracy = " + str(accuracy)
                  )
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(name + ".pdf", bbox_inches='tight')
    return ax


#############################################

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

# define the model
def define_model(length, vocab_size):
	# channel 1
	inputs1 = layers.Input(shape=(length,))
	embedding1 = layers.Embedding(vocab_size, 100)(inputs1)
	conv1 = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
	drop1 = layers.Dropout(0.5)(conv1)
	pool1 = layers.MaxPooling1D(pool_size=2)(drop1)
	flat1 = layers.Flatten()(pool1)
	# channel 2
	inputs2 = layers.Input(shape=(length,))
	embedding2 = layers.Embedding(vocab_size, 100)(inputs2)
	conv2 = layers.Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
	drop2 = layers.Dropout(0.5)(conv2)
	pool2 = layers.MaxPooling1D(pool_size=2)(drop2)
	flat2 = layers.Flatten()(pool2)
	# channel 3
	inputs3 = layers.Input(shape=(length,))
	embedding3 = layers.Embedding(vocab_size, 100)(inputs3)
	conv3 = layers.Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = layers.Dropout(0.5)(conv3)
	pool3 = layers.MaxPooling1D(pool_size=2)(drop3)
	flat3 = layers.Flatten()(pool3)
	# merge
	merged = layers.concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = layers.Dense(10, activation='relu')(merged)
	outputs = layers.Dense(2, activation='sigmoid')(dense1)
	model = tf.keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize
	print(model.summary())
	plot_model(model, show_shapes=True, to_file='multichannel.png')
	return model

# load training dataset
trainLines, trainLabels = load_dataset('train.pkl')

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
print(trainX.shape)

# define model
model = define_model(length, vocab_size)

# fit model
model.fit([trainX,trainX,trainX], array(trainLabels), epochs=8, batch_size=16)
# save the model
model.save('model.h5')

# evaluate model on training dataset
loss, acc = model.evaluate([trainX, trainX, trainX], array(trainLabels), verbose=0)
print('Train Accuracy: %f' % (acc * 100))



testLines, testLabels = load_dataset('test.pkl')
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
print(trainX.shape, testX.shape)


# evaluate model on test dataset dataset
loss, acc = model.evaluate([testX, testX, testX], array(testLabels), verbose=0)
print('Test Accuracy: %f' % (acc * 100))

y_pred = model.predict([testX, testX, testX], verbose=0)

reverse_mapping = {0: "neg", 1: "pos"}

y_pred = [reverse_mapping[np.argmax(y) ]for y in y_pred]

y_test = [reverse_mapping[np.argmax(y)] for y in testLabels]



labels = ["neg", "pos"]
title = "CNN Confusion Matrix"
ax = evaluate_model(y_test, y_pred, labels, title=title, name="CNN", width=7, height=5)
plt.show()