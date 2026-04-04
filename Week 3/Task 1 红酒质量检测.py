import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.mean = None
        self.std = None
        self.theta = None

    def train_test_split(self, test_size = 0.2, random_state = 42):
        np.random.seed(random_state)
        indexes = np.random.permutation(len(self.x))
        split_index = int(test_size * len(indexes))
        x_train_index = indexes[split_index:]
        x_test_index = indexes[:split_index]
        x_train = self.x[x_train_index]
        x_test = self.x[x_test_index]
        y_train = self.y[x_train_index]
        y_test = self.y[x_test_index]

        return x_train, x_test, y_train, y_test

    def StandardScaler(self,x_train, x_test):
        self.mean = x_train.mean(axis=0)
        self.std = x_train.std(axis=0)
        self.std[self.std == 0] = 1

        x_train_scaled = (x_train - self.mean) / self.std
        x_test_scaled = (x_test - self.mean) / self.std

        x_train_scaled = np.c_[np.ones(x_train_scaled.shape[0]), x_train_scaled]
        x_test_scaled = np.c_[np.ones(x_test_scaled.shape[0]), x_test_scaled]

        return x_train_scaled, x_test_scaled

    def fit(self, x_train_scaled, y_train):
        self.theta = np.linalg.inv(x_train_scaled.T @ x_train_scaled) @ x_train_scaled.T @ y_train

    def predict(self, x_test_scaled):
        return x_test_scaled @ self.theta

    def score(self, y_pre, y_test):
        MSE = np.mean((y_pre - y_test) ** 2)

        ss_res = np.sum((y_test - y_pre) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        R2 = 1 - ss_res / ss_tot

        return MSE, R2

class LogisticRegression:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.mean = None
        self.std = None
        self.theta = None

    def train_test_split(self, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        indexes = np.random.permutation(len(self.x))
        split_index = int(test_size * len(indexes))
        x_train_index = indexes[split_index:]
        x_test_index = indexes[:split_index]
        x_train = self.x[x_train_index]
        x_test = self.x[x_test_index]
        y_train = self.y[x_train_index]
        y_test = self.y[x_test_index]

        return x_train, x_test, y_train, y_test

    def StandardScaler(self,x_train, x_test):
        self.mean = x_train.mean(axis=0)
        self.std = x_train.std(axis=0)
        self.std[self.std == 0] = 1

        x_train_scaled = (x_train - self.mean) / self.std
        x_test_scaled = (x_test - self.mean) / self.std

        x_train_scaled = np.c_[np.ones(x_train_scaled.shape[0]), x_train_scaled]
        x_test_scaled = np.c_[np.ones(x_test_scaled.shape[0]), x_test_scaled]

        return x_train_scaled, x_test_scaled

    def Sigmoid(self, y_pre):
        return 1 / (1 + np.exp(-y_pre))

    def gradient(self, x_train_scaled, y_train, theta):
        h = self.Sigmoid(x_train_scaled @ theta)
        return (x_train_scaled.T @ (h - y_train)) / len(y_train)

    def fit(self, x_train_scaled, y_train,a = 0.1, epochs = 1000, tol = 1e-6):
        self.theta = np.zeros(x_train_scaled.shape[1])

        for i in range(epochs):
            grad = self.gradient(x_train_scaled, y_train, self.theta)
            theta_new = self.theta - a * grad
            if np.linalg.norm(theta_new - self.theta) < tol:
                break
            self.theta = theta_new

    def predict(self, x_test_scaled):
        return self.Sigmoid(x_test_scaled @ self.theta)

    def score(self, y_pre, y_test):
        tp = np.sum((y_pre == 1) & (y_test == 1))
        tn = np.sum((y_pre == 0) & (y_test == 0))
        fp = np.sum((y_pre == 1) & (y_test == 0))
        fn = np.sum((y_pre == 0) & (y_test == 1))

        accuracy = (tp + tn) / len(y_test)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        conf_matrix = np.array([[tn, fp],
                                [fn, tp]])
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix
        }


wine_df = pd.read_csv('E:\\pycharm项目\\QG\\data\\wine+quality\\winequality-red.csv', sep = ';')

x = wine_df.drop('quality', axis = 1)
y = wine_df['quality']
y_binary = (wine_df['quality'] > 6).astype(int)

# 线性回归模型
wine_reg = LinearRegression(x, y)

x_train, x_test, y_train, y_test = wine_reg.train_test_split(test_size = 0.2, random_state = 42)

x_train_scaled, x_test_scaled = wine_reg.StandardScaler(x_train, x_test)

wine_reg.fit(x_train_scaled, y_train)

y_pre = wine_reg.predict(x_test_scaled)

Mse, R2 = wine_reg.score(y_pre, y_test)

print(f"线性回归模型评估：\nMSE: {Mse}\nR2: {R2}")

# 逻辑回归模型
wine_log = LogisticRegression(x, y_binary)

x_train, x_test, y_train, y_test = wine_log.train_test_split(test_size = 0.2, random_state = 42)

x_train_scaled, x_test_scaled = wine_log.StandardScaler(x_train, x_test)

wine_log.fit(x_train_scaled, y_train)

y_pre_odds = wine_log.predict(x_test_scaled)

log_score = wine_log.score(y_pre_odds, y_test)

print(f"逻辑回归模型：\n{log_score}")
