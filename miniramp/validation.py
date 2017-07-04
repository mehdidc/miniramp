from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

def kfold(X, y, n_splits=3, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return kf.split(X, y)


def shuffle_split(X, y, n_splits=3, test_size=0.3, random_state=42):
    kf = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    return kf.split(X, y)
