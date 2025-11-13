import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("stock_data.csv")

# Features and target
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict for new value
new_data = [[1500, 1520, 1490, 3000000]]
predicted_price = model.predict(new_data)
print("Predicted Stock Price:", predicted_price[0])
