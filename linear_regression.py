import pandas as pd


class MyLineReg():

    def __init__(self, n_iter, learning_rate, weights=None):
        if weights is None:
            weights = []
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose:int=False):
        123