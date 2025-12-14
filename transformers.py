import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CoerceNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X