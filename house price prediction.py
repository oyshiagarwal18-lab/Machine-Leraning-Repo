import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Sample dataset
data = {
    "Area": [1000, 1500, 1800, 2400, 3000, 3500, 4000],
    "Bedrooms": [2, 3, 3, 4, 4, 5, 5],
    "Price": [200000, 300000, 350000, 450000, 500000, 600000, 650000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
# Accuracy check
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

# Predict new house price
new_house = [[2000, 3]]  # area, bedrooms
predicted_price = model.predict(new_house)

print("Predicted Price:", predicted_price[0])
''