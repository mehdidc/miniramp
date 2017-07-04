from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss as sk_log_loss

def accuracy(clf, X, y):
    y_pred = clf.predict(X)
    return (y == y_pred).mean()


def auc(clf, X, y):
    y_pred = clf.predict(X)
    return roc_auc_score(y, y_pred)


def log_loss(clf, X, y):
    y_pred = clf.predict_proba(X)
    return sk_log_loss(y, y_pred)
