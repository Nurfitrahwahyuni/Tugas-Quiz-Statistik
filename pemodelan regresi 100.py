import pandas as pd
import numpy as np

data = pd.read_csv ('winequality-red.csv')

# Ambil 100 data
sample_data = data.sample(100)

# Ambil variabel numerik
X = sample_data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides']]
Y = sample_data['quality']

# Hitung koefisien regresi menggunakan numpy
coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
print("Koefisien regresi:\n", coefficients)