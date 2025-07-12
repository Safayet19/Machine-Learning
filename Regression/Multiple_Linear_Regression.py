import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Dataset with one missing height value
data = {
    'Age': [25, 30, 35, 40, 45],
    'Height': [162.56, 172.72, 167.64, np.nan, 157.48],  # missing value
    'Weight': [70, 95, 78, 110, 85],
    'Premium': [18000, 38000, 38000, 60000, 70000]
}

dataset = pd.DataFrame(data)

# Step 1: Handle missing value in 'Height'
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset['Height'] = imputer.fit_transform(dataset[['Height']])

# Step 2: Define inputs (X) and output (y)
X = dataset[['Age', 'Height', 'Weight']]
y = dataset['Premium']

# Step 3: Create and train the model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict premium for new values
# Example: Predict for Age=40, Height=170, Weight=90
new_data = np.array([[40, 170, 90]])
prediction = model.predict(new_data)

print("Predicted Premium for Age=40, Height=170, Weight=90:", prediction[0])
