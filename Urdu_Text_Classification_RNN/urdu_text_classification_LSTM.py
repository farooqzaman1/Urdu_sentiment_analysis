# LSTM for sequence classification in the IMDB dataset
import os
from data_2 import supervised, vectorized

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.utils.vis_utils import plot_model

import numpy
import tensorflow  as  tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing import sequence

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from matplotlib import pyplot as plt
import seaborn as sns

# fix random seed for reproducibility
numpy.random.seed(7)

top_words = 16000
max_review_length = 200
embedding_vecor_length = 128


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
mapp = {"neg": 0, "pos": 1}
reverse_mapping = {0: "neg", 1: "pos"}

# X2, y2 = supervised()

X2, y2 = vectorized(mapp)

print(y2.shape)

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.33, random_state=42)

############################################33
## truncate and pad input sequences

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

model = tf.keras.Sequential()
model.add(layers.Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(layers.LSTM(15))
model.add(layers.Dense(2, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
###plot_model(model, show_shapes=True, to_file='multichannel.png')
model.fit(X_train, y_train, epochs=5, batch_size=64)
##############################

# def fit_lstm(X2, y2, batch_size, nb_epoch, neurons):
#     model = tf.keras.Sequential()
#     model.add(layers.LSTM(neurons, batch_input_shape=(batch_size, X2.shape[1], X2.shape[2]), stateful=True))
#     model.add(layers.Dense(3))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     for i in range(nb_epoch):
#         model.fit(X2, y2, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
#         model.reset_states()
#     return model


##########################
# model = fit_lstm(X2, y2, batch_size=1, nb_epoch=8,  neurons=15)
print("\nEvaluating....")
scores = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict_classes(X_test)

print("Accuracy: %.2f%%" % (scores[1] * 100))


y_classes_test = np.array([reverse_mapping[np.argmax(y, axis=None, out=None)] for y in y_test])
y_pred = np.array([reverse_mapping[y] for y in y_pred])

labels = ["neg", "pos"]
title = "LSTM Confusion Matrix"
ax = evaluate_model(y_classes_test, y_pred, labels, title=title, name="LSTM", width=7, height=5)
plt.show()
