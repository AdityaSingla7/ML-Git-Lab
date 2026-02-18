import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler # Added for scaling

# 1. Load dataset
data = pd.read_csv('dataset.csv')
X = data[['SquareFeet', 'Bedrooms']]
y = data['Price']

# 2. Feature Scaling (The change for Part 3)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train model on scaled data
model = LinearRegression()
model.fit(X_scaled, y)

# 4. Print accuracy
accuracy = model.score(X_scaled, y)
print(f"Model Training Accuracy (Scaled): {accuracy}")
