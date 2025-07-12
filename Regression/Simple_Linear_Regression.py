import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (Age vs Premium)
data = {
    'Age': [25, 30, 35, 40, 45],
    'Premium': [18000, 32000, 42000, 47000, 55000]
}
df = pd.DataFrame(data)

# Independent and dependent variables
X = df[['Age']]     # Make sure X is 2D
y = df['Premium']   # y can be 1D

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict for new value (optional example)
predicted = model.predict([[50]])  # Predict premium for age 50
print("Predicted Premium for Age 50:", predicted[0])

# Create smooth line for plotting
x_grid = np.linspace(df['Age'].min(), df['Age'].max(), 100).reshape(-1, 1)
y_grid = model.predict(x_grid)

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(x_grid, y_grid, color='green', label='Linear Regression Line')

plt.title('Age vs Premium (Simple Linear Regression)')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.legend()
plt.grid(True)
plt.show()
