import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

data = pd.read_csv ('winequality-red.csv')

x = data [['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y = data[['quality']]

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

standardized_residuals = model.get_influence().resid_studentized_internal

plt.scatter(model.predict(), standardized_residuals, alpha=0.5)
plt.title('Scatterplot Prediksi vs. Residual Standar')
plt.xlabel('Nilai Prediksi')
plt.ylabel('Residual Standar')

bp_test_statistic, bp_p_value, _, _ = het_breuschpagan(standardized_residuals, x)
print(f"Breusch-Pagan Test Statistic: {bp_test_statistic}")
print(f"P-value (Breusch-Pagan Test): {bp_p_value}")

plt.show()