import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, mean_squared_error, mean_absolute_error, accuracy_score, r2_score
from sklearn.metrics import auc
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv("processed_diadetes.csv")
df2 = pd.read_csv("processed_Fish.csv")


X = df2[['Weight']]
y = df2['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
DecisionTreeRegressor(criterion='gini')

dt_regressor = DecisionTreeRegressor(max_depth=5,min_samples_split=10,min_samples_leaf=5,random_state=42 )
dt_regressor.fit(X_train, y_train)

dt_regressor.fit(X_train, y_train)

y_pred = dt_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE: ", mse)
print("MAE: ", mae)
print("R^2", r2_score(y_test, y_pred),  '\n')
X1 = df.drop(['Outcome'], axis=1)
y1 = df['Outcome']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.4, random_state=42)

dt_classifier = DecisionTreeClassifier(max_depth=4,min_samples_split=10,min_samples_leaf=5,random_state=42)

dt_classifier.fit(X1_train, y1_train)

y1_proba = dt_classifier.predict_proba(X1_test)

y1_pred_test = dt_classifier.predict(X1_test)
accuracy = accuracy_score(y1_test, y1_pred_test)
print("Точность: " , accuracy)

fpr, tpr, thresholds = roc_curve(y1_test, y1_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(1, figsize=(8, 6))
plt.plot(fpr, tpr, marker='o')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1])
plt.ylim([0.0, 1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.grid(True)

plt.figure(2, figsize=(14, 8))
tree.plot_tree(dt_regressor)
plt.title('Decision Tree Regressor')

plt.figure(3, figsize=(14, 8))
tree.plot_tree(dt_classifier)
plt.show()