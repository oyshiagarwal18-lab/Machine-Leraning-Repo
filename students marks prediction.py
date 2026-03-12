# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset (study hours vs marks)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([40, 50, 60, 70, 80])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict marks for 6 hours study
prediction = model.predict([[6]])

print("Predicted Marks:", prediction)
