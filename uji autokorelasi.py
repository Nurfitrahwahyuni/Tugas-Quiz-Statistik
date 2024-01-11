import pandas as pd
import statsmodels.api as sm

data = pd.read_csv ('winequality-red.csv')

x = data [['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y = data[['quality']]

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

durbin_watson_statistic = sm.stats.durbin_watson(model.resid)

if durbin_watson_statistic < 1.5:
    interpretation = "Autokorelasi positif (residuals cenderung positif)"
elif durbin_watson_statistic > 2.5:
    interpretation = "Autokorelasi negatif (residuals cenderung negatif)"
else:
    interpretation = "Tidak ada autokorelasi yang signifikan"

print(f"\nDurbin-Watson Statistic: {durbin_watson_statistic}")
print(f"Interpretasi: {interpretation}")

print(model.summary())