import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # градиентный спуск
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred > 0.5).astype(int)

# чтение данных
data = pd.read_csv('diabetes.csv', delimiter='\t')

# разделение данных на признаки и метки
X = data.drop(columns=['Диагноз']).values
y = data['Диагноз'].values

# разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# создание и обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# предсказание меток для тестовых данных
y_pred = model.predict(X_test)

# оценка точности классификации
accuracy = accuracy_score(y_test, y_pred)
print("Точность классификации для исходных данных:", accuracy)

# вычисление матрицы корреляции
correlation_matrix = data.corr().abs()

# выбор верхнего треугольника матрицы корреляции (без диагонали)
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# нахождение признаков с высокой корреляцией
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.4) and column != 'Диагноз']

# удаление признаков с высокой корреляцией из данных
data_filtered = data.drop(columns=high_corr_features)

# выделение новой матрицы признаков и вектора меток
X_filtered = data_filtered.drop(columns=['Диагноз']).values
y_filtered = data_filtered['Диагноз'].values

# вывод столбцов, которые были удалены с помощью CFS
removed_columns = set(data.columns) - set(data_filtered.columns)
print("Удаленные столбцы:", removed_columns)

# разделение отфильтрованных данных на обучающую и тестовую выборки
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# создание и обучение новой модели на отфильтрованных данных
model_filtered = LogisticRegression()
model_filtered.fit(X_train_filtered, y_train_filtered)

# предсказание меток для тестовых данных после отбора признаков
y_pred_filtered = model_filtered.predict(X_test_filtered)

# оценка точности классификации после отбора признаков
accuracy_filtered = accuracy_score(y_test_filtered, y_pred_filtered)
print("Точность классификации после отбора признаков:", accuracy_filtered)