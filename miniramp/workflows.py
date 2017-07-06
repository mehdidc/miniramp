import time

from collections import defaultdict

from .utils import build_func
from .utils import base_name
from .utils import eval_and_get
from .utils import import_object
from .utils import ClassifierEnsemble

def classification(codes, data, validation, scores, options):
    final_model_strategy = options['final_model_strategy']
    assert final_model_strategy in ('retrain', 'bagging')

    Classifier = eval_and_get(codes['classifier'], 'Classifier')
    load_data = build_func(data)
    split = build_func(validation)
    score_funcs = {base_name(score): build_func(score) for score in scores}

    (X_train_full, y_train_full), (X_test, y_test) = load_data()
    
    train_scores = defaultdict(list)
    valid_scores = defaultdict(list)
    clfs = []
    for train, valid in split(X_train_full, y_train_full):
        X_train = X_train_full[train]
        y_train = y_train_full[train]
        X_valid = X_train_full[valid]
        y_valid = y_train_full[valid]
        
        t0 = time.time()
        clf = Classifier()
        clf.fit(X_train, y_train)
        clfs.append(clf)
        train_scores['time'].append(time.time() - t0)
        
        t0 = time.time()
        for name, func in score_funcs.items():
            train_scores[name].append(func(clf, X_train, y_train))
            valid_scores[name].append(func(clf, X_valid, y_valid))
        valid_scores['time'].append(time.time() - t0)
        
    if final_model_strategy == 'retrain':
        t0 = time.time()
        clf = Classifier()
        clf.fit(X_train_full, y_train_full)
        train_full_scores['time'] = time.time() - t0
    elif final_model_strategy == 'bagging':
        clf = ClassifierEnsemble(clfs)
    
    train_full_scores = {}
    test_scores = {}

    t0 = time.time()
    for name, func in score_funcs.items():
        train_full_scores[name] = func(clf, X_train_full, y_train_full)
        test_scores[name] = func(clf, X_test, y_test)
    test_scores['time'] = time.time() - t0

    out = {
        'train': train_scores,
        'valid': valid_scores,
        'train_full': train_full_scores,
        'test': test_scores
    }
    return out 
classification.requirements = ['classifier']
