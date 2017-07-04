import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .utils import get_file

def read_csv(filename, y_col, test_size=0.3, random_state=42):
    filename = get_file(os.path.basename(filename), filename)
    df = pd.read_csv(filename)
    X = df.drop(y_col, axis=1).values
    y = df[y_col].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
