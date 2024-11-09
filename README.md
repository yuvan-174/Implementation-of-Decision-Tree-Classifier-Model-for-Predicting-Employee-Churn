# Ex 08 - Implementation of Decision Tree Classifier Model for Predicting Employee Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Split the dataset into features (X) and target (y), and preprocess the data.
2. Split data into training and testing sets.
3. Train the Decision Tree Classifier using the training data.
4. Predict and evaluate the model on the test data, then visualize the decision tre

## Program:
```py
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: YUVAN SUNDAR S
RegisterNumber:  212223040250
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load dataset
file_path = 'Employee.csv'
data = pd.read_csv(file_path)

# Preprocessing: Convert categorical variables to numerical
data = pd.get_dummies(data)

# Split features and target ('left' is the churn indicator)
X = data.drop('left', axis=1)
y = data['left']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Stayed', 'Left'], filled=True)
plt.show()

```

## Output:
![Screenshot 2024-09-30 154913](https://github.com/user-attachments/assets/1712985d-38bf-4422-b5fa-4be72e7ca4c2)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
