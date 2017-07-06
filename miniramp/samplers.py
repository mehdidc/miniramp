import numpy as np
rng = np.random

classifier = """
{head}
class Classifier:
    
    def __init__(self):
        self.clf = {clf}

    def fit(self, X, y):
        X = X.reshape((X.shape[0], -1))
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        X = X.reshape((X.shape[0], -1))
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = X.reshape((X.shape[0], -1))
        return self.clf.predict_proba(X)
"""

def _rf_classifier():
    n_estimators = rng.randint(1, 100)
    max_depth = rng.randint(1, 50)
    code = classifier.format(
        head="from sklearn.ensemble import RandomForestClassifier",
        clf="RandomForestClassifier(max_depth={}, n_estimators={})".format(max_depth, n_estimators)
    )
    return code, {'n_estimators': n_estimators, 'max_depth': max_depth}


def sklearn_classifier():
    code, hypers = _rf_classifier()
    code = code.format(**hypers)
    out = {
        'codes': {
            'classifier': code
        },
        'info': hypers
    }
    return out


keras_classifier = """
import numpy as np
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
EPS = 1e-7
class Classifier:
    
    def fit(self, X, y):
        axes = tuple(set(range(len(X.shape))) - set([1]))
        self.mu = X.mean(axis=axes, keepdims=True)
        self.std = X.std(axis=axes, keepdims=True)
        X = (X - self.mu) / (self.std + EPS)
        n_outputs = len(np.unique(y))
        X = X.reshape((X.shape[0], -1))
        inp = Input((X.shape[1],))
        x = inp
        n_hiddens = {layers}
        for n_hidden in n_hiddens:
            x = Dense(n_hidden, activation={activation})(x)
        out = Dense(n_outputs, activation='softmax')(x)
        self.model = Model(inp, out)
        self.model.summary()
        opt = Adam(lr=1e-3)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        y = to_categorical(y, n_outputs)
        self.model.fit(X, y)
        self.model.predict_proba = self.model.predict
        self.model.predict = lambda X: self.model.predict_proba(X).argmax(axis=1)
        
    def predict(self, X):
        X = (X - self.mu) / (self.std + EPS)
        X = X.reshape((X.shape[0], -1))
        return self.model.predict(X)

    def predict_proba(self, X):
        X = (X - self.mu) / (self.std + EPS)
        X = X.reshape((X.shape[0], -1))
        return self.model.predict_proba(X)
"""

def keras_dense_classifier():
    n_layers = _sample_nb_layers()
    layers = [_sample_size_dense_layer() for _ in range(n_layers)]
    layers_s = '[' + (','.join(map(str, layers)))  + ']'
    activation = _sample_activation()
    code = keras_classifier.format(
        layers=layers_s, 
        activation="'" + activation + "'",
    )
    hypers = {
        'layers': layers,
        'activation': activation
    }
    out = {
        'codes': {
            'classifier': code
        },
        'info': hypers
    }
    return out


def _sample_nb_layers():
    return 1


def _sample_size_dense_layer():
    return 100

def _sample_activation():
    return 'relu'
