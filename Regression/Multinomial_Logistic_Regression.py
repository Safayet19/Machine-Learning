# Step 1: Import libraries
import pandas as pd                # For handling dataset as DataFrame (tabular data)
import numpy as np                 # For numerical operations
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.model_selection import train_test_split # To split dataset into training and testing
from sklearn.preprocessing import LabelEncoder       # To convert labels (species) into numbers
from sklearn.metrics import classification_report, confusion_matrix  # For model evaluation
import matplotlib.pyplot as plt    # For plotting charts
import seaborn as sns              # For nicer data visualizations (confusion matrix heatmap)

# Step 2: Load dataset
dataset = pd.read_csv('iris.csv')

# Step 3: Select input features and target (what we want to predict)
x = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  # Independent variables
y = dataset['Species']  # Dependent variable (target classes: Iris-setosa, Iris-versicolor, Iris-virginica)

# Step 4: Encode target labels (string to numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Converts species names to numbers 

# Step 5: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=0)

# Step 6: Create and train the Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# Step 7: Predict on test data
y_pred = model.predict(X_test)

# Step 8: Convert encoded predictions back to original species names
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# Step 9: Show classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)

# Step 11: Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Multinomial Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Pairplot with hue set to species to show different classes in different colors
sns.pairplot(dataset, hue='Species', diag_kind='kde', markers=["o", "s", "D"])

plt.suptitle("Pairplot of Iris Dataset", y=1.02)  # title with some padding
plt.show()