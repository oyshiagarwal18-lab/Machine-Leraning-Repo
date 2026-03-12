# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Example dataset
data = {
    'Vehicles': [100, 150, 200, 250, 300],
    'Speed': [40, 50, 60, 70, 80],
    'Accidents': [5, 7, 12, 15, 20]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Vehicles', 'Speed']]
y = df['Accidents']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict accidents
prediction = model.predict([[220, 65]])

print("Predicted number of accidents:", prediction)
