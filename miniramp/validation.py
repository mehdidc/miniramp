from sklearn.model_selection import KFold

def kfold(X, y, n_splits=3, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return kf.split(X, y)
