# from google.colab import files
# uploaded = files.upload()

# import zipfile
# import io
# zip_file = list(uploaded.keys())[0]
# with zipfile.ZipFile(io.BytesIO(uploaded[zip_file]), 'r') as zip_ref:
#     zip_ref.extractall('house-prices-data')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('house-prices-data/train.csv')
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

example = pd.DataFrame({'GrLivArea': [2000], 'BedroomAbvGr': [3], 'FullBath': [2]})
predicted_price = model.predict(example)
print(f'Predicted price for example house: ${predicted_price[0]:.2f}')
coefficients = pd.Series(model.coef_, index=features)
print("\nModel Coefficients:")
print(coefficients)
print(f"\nIntercept: {model.intercept_:.2f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")