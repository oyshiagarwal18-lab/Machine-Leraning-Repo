# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([[1], [2], [3], [4], [5]])   # independent variable
y = np.array([2, 4, 6, 8, 10])            # dependent variable

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict new value
prediction = model.predict([[6]])

print("Predicted value:", prediction)
