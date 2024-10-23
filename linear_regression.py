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
        X = pd.concat([pd.DataFrame([[1]] * len(X)), X], axis=1)
        X.columns = range(X.shape[1])
        self.weights = [1] * (X.shape[1] + 1)

        for i in range(self.n_iter):
            predictions = [sum(list(Xes) * self.weights) for Xes in X.iterrows()]

            loss = 1/len(X) * sum((predictions - y)**2)

            if verbose:
                print(f"Iteration {i+1}/{self.n_iter}, Loss: {loss}")

regression = MyLineReg(1000,0.2)
X = pd.DataFrame([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
y = pd.Series([1,2,3])
regression.fit(X,y)