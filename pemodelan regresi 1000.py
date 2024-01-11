User
import pandas as pd
import numpy as np

data = pd.read_csv ('winequality-red.csv')

# Ambil 100 data
sample_data = data.sample(100)

# Ambil variabel numerik
X_all = data [['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
Y_all = data[['quality']]

# Hitung koefisien regresi menggunakan numpy
coefficients_all = np.linalg.inv(X_all.T @ X_all) @ X_all.T @ Y_all
print("Koefisien regresi (seluruh data):", coefficients_all)