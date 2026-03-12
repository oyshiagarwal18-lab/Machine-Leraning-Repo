import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
# [Study hours, Sleep hours]
X = np.array([
    [2,6],
    [3,7],
    [4,8],
    [5,6],
    [6,7]
])

# Output (Marks)
y = np.array([50,55,65,70,75])

# Model
model = LinearRegression()

# Training
model.fit(X, y)

# Prediction
result = model.predict([[4,7]])

print("Predicted Marks:", result)