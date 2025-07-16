# Step 1: Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 2: Load the dataset manually into a DataFrame
# data = pd.read_csv(r"E:\3rd year\AI & ML\Decision_Treee.csv")
data = pd.DataFrame({
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny',
                'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'temp': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild',
             'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high',
                 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'windy': [False, True, False, False, False, True, True, False,
              False, False, True, True, False, True],
    'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no',
             'yes', 'yes', 'yes', 'yes', 'yes', 'no']
})

# Step 3: Convert categorical columns to numerical values using Label Encoding
# Since DecisionTreeClassifier requires numerical input
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])# Step 3: Convert categorical columns to numerical values using Label Encoding
# Since DecisionTreeClassifier requires numerical input
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Step 4: Split features (X) and target (y)
X = data.drop('play', axis=1)  # All columns except 'play'
y = data['play']               # Target column

# Step 5: Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 6: Create and train the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy')  # or use 'gini'
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred)) #Accuracy: 0.6666666666666666

# Step 9: Visualize the Tree
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)

plt.figure(figsize=(12, 6))
tree.plot_tree(classifier, feature_names=['outlook', 'temp', 'humidity', 'windy'],
               class_names=['no', 'yes'], filled=True)
plt.show()
