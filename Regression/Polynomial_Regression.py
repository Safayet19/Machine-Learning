import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Your dataset
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager', 'Country Manager',
                 'Region Manager', 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1,2,3,4,5,6,7,8,9,10],
    'Salary': [45000,50000,60000,80000,110000,150000,200000,300000,500000,1000000]
}
dataset = pd.DataFrame(data)

# Independent and dependent variables
X = dataset[['Level']]  # DataFrame for sklearn
y = dataset['Salary']

# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Polynomial Regression model (degree 4)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Smooth range for plotting
x_grid = np.arange(X['Level'].min(), X['Level'].max() + 0.1, 0.1).reshape(-1,1)

# Predictions
y_linear = linear_model.predict(x_grid)
y_poly = poly_model.predict(poly.transform(x_grid))

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(x_grid, y_linear, color='green', label='Linear Regression')
plt.plot(x_grid, y_poly, color='red', label='Polynomial Regression')

plt.title('Position Level vs Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()