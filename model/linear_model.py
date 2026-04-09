import numpy as np
from model.utils import get_train_val, plot_learning_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, r2_score

def initialize_weight(dim):
    W0 = np.array([[0]])  # bias
    W = np.random.rand(dim, 1)
    return np.concatenate((W0, W))


class LinearModel(BaseEstimator, ClassifierMixin):
    def __init__(self, dim=None, is_reg=False, loss_fn=None, grad_fn=None, act_fn=None,
                 lr=0.1, reg_type='', reg_lambda=0, n_iteration=50, val_ratio=.2,
                 random_state=None, verbose=True, plot_curve=True):

        self.dim = dim
        self.is_reg = is_reg
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.act_fn = act_fn if act_fn is not None else (lambda x: x)

        self.lr = lr
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.n_iteration = n_iteration
        self.val_ratio = val_ratio

        self.random_state = random_state
        self.verbose = verbose
        self.plot_curve = plot_curve

        if self.dim is not None:
            self.W = initialize_weight(self.dim)
        else:
            self.W = None

        self.train_losses = []
        self.val_losses = []
        self.loss_history = []  # unified history for plotting


    def _ensure_bias_column(self, X):
        if X.shape[1] == self.dim:
            X0 = np.ones((X.shape[0], 1))
            return np.concatenate((X0, X), axis=1)

        if X.shape[1] == self.dim + 1:
            return X

        raise ValueError(f'Input feature dimension mismatch: expected {self.dim} or {self.dim + 1}, got {X.shape[1]}')


    def fit(self, X, y, lr=None, reg_type=None, reg_lambda=None,
            n_iteration=None, val_ratio=None):

        if self.dim is None:
            self.dim = X.shape[1]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # always reinitialize weights (as in template behavior)
        self.W = initialize_weight(self.dim)

        self.train_losses = []
        self.val_losses = []
        self.loss_history = []

        lr = self.lr if lr is None else lr
        reg_type = self.reg_type if reg_type is None else reg_type
        reg_lambda = self.reg_lambda if reg_lambda is None else reg_lambda
        n_iteration = self.n_iteration if n_iteration is None else n_iteration
        val_ratio = self.val_ratio if val_ratio is None else val_ratio

        X = self._ensure_bias_column(X)
        X_train, y_train, X_val, y_val = get_train_val(X, y, val_ratio)

        for i in range(n_iteration):

            # forward
            y_pred = self.act_fn(np.squeeze(X_train @ self.W))

            # loss
            loss = self.loss_fn(y_train, y_pred)
            self.train_losses.append(loss)
            self.loss_history.append(loss)

            # gradient
            grad = self.grad_fn(y_train, y_pred)
            grad_w = X_train.T @ grad

            if len(grad_w.shape) == 1:
                grad_w = grad_w[:, None]

            # L2 regularization (exclude bias!)
            if reg_type == 'l2':
                grad_w[1:] += 2 * (reg_lambda / len(X_train)) * self.W[1:]

            # update
            self.W -= lr * grad_w

            # validation
            y_val_pred = self.act_fn(np.squeeze(X_val @ self.W))
            val_loss = self.loss_fn(y_val, y_val_pred)
            self.val_losses.append(val_loss)

            if self.verbose and (i + 1) % 50 == 0:
                print(f'{i+1}. Training loss: {loss:.4f}, Val loss: {val_loss:.4f}')

        if self.plot_curve:
            plot_learning_curve(self.train_losses, self.val_losses)

        return self


    def get_weight(self):
        return self.W


    def predict(self, X, thres=0.5):
        if X.shape[1] == self.dim:
            X0 = np.ones((X.shape[0], 1))
            X = np.concatenate((X0, X), axis=1)

        y_pred = self.act_fn(np.squeeze(X @ self.W))

        if not self.is_reg:
            y_pred = (y_pred >= thres).astype(np.uint8)

        return y_pred


    def predict_proba(self, X):
        if self.is_reg:
            raise Exception('Cannot predict probability for regression')

        if X.shape[1] == self.dim:
            X0 = np.ones((X.shape[0], 1))
            X = np.concatenate((X0, X), axis=1)

        return self.act_fn(np.squeeze(X @ self.W))


    def score(self, X, y):
        y_pred = self.predict(X)

        if self.is_reg:
            return r2_score(y, y_pred)

        return accuracy_score(y, y_pred)