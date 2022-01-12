from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from keras.utils import to_categorical
import pandas as pd
import numpy as np


def process_prepare_data():
    data1 = pd.read_excel("data_files/dramas.xlsx")
    data2 = pd.read_excel("data_files/Food and receip.xlsx")
    data3 = pd.read_excel("data_files/Politics.xlsx")
    data4 = pd.read_excel("data_files/software blog and forum reviews.xlsx")
    data5 = pd.read_excel("data_files/sports.xlsx")
    frames = [data1, data2, data3, data4, data5]
    data = pd.concat(frames, sort=False)

    data_annotated = {}
    for row in data.values:
        review = row[3]
        a1, a2, a3 = row[5], row[6], row[7]
        if (a2 is np.nan):
            print("skipping....")
            continue
        else:
            data_annotated[review] = {'pos': 0, 'neg': 0, 'neu': 0}
            data_annotated[review][a1] += 1
            data_annotated[review][a2] += 1
            data_annotated[review][a3] += 1

    data_annotated_sorted = {}
    for k, v in data_annotated.items():
        data_annotated_sorted[k] = {k1: v1 for k1, v1 in sorted(v.items(), key=lambda item: item[1], reverse=True)}

    data_final = {}
    for k, v in data_annotated_sorted.items():
        data_final[k] = list(v)[0]
    data_final_df = pd.DataFrame(list(data_final.items()))
    data_final_df.to_excel("data.xlsx")
    return  data_final_df

def get_data_tf_vector():
    data = pd.read_excel("data1.xlsx")
    documents = data['Text'].values
    y = data['Label'].values
    le = preprocessing.LabelEncoder()
    y = le.fit(y).transform(y)
    y = to_categorical(y, num_classes=3, dtype='float32')
    vectorizer = CountVectorizer(max_features=4000, min_df=5, max_df=0.7)
    X = vectorizer.fit_transform(documents).toarray()
    return X, y


def read_data():
    data = pd.read_excel("data_binary.xlsx")
    from sklearn.utils import shuffle
    data = shuffle(data)
    X = data['Text'].values
    y = data['Label'].values
    return X, y

def Label_encoder_catagorical(y, mapping):
    y = np.array([mapping[i] for i in y])
    clases = len(mapping)
    y = to_categorical(y, num_classes=clases, dtype='float32')
    return y