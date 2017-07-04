import numpy as np
rng = np.random

rf_classifier = """
from functools import partial
from sklearn.ensemble import RandomForestClassifier
Classifier = partial(RandomForestClassifier,
    max_depth={max_depth}, 
    n_estimators={n_estimators}
)
"""
def _rf_classifier():
    n_estimators = rng.randint(1, 100)
    max_depth = rng.randint(1, 50)
    return rf_classifier, {'n_estimators': n_estimators, 'max_depth': max_depth}


def classifier():
    code, hypers = _rf_classifier()
    code = code.format(**hypers)
    out = {
        'codes': {
            'classifier': code
        },
        'info': hypers
    }
    return out
