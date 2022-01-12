from pandas import DataFrame
from pandas import concat
from data import read_data, Label_encoder_catagorical
import numpy as np


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def vectorized(mapp):
    X, y = read_data()
    corpus = {}
    for row in X:
        for w in row.split():
            if w not in corpus:
                corpus[w] = len(corpus)
    X1 = []
    for sent in X:
        vec = []
        for w in sent.split():
            if w in corpus:
                vec.append(corpus[w])
        X1.append(vec)
    X1 = np.array(X1)

    y1 = Label_encoder_catagorical(y, mapping=mapp)
    return X1, y1



def supervised():
    X1, y1 = vectorized()
    raw = DataFrame()
    raw['Text'] = X1
    raw['Label'] = list(y1)
    values = raw.values
    data = series_to_supervised(values, 1, 2)
    supervised_values = data.values
    # split data into train and test-sets
    X2, y2 = supervised_values[:, 0:-1], supervised_values[:, -1]
    X2 = X2.reshape(X2.shape[0], 1, X2.shape[1])
    y2 = [list(i) for i in y2]
    y2 = np.array(y2)
    return X2, y2

