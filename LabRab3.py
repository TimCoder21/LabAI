import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, mean_squared_error, mean_absolute_error
from sklearn.metrics import auc
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

df = pd.read_csv("processed_titanic.csv")

X = df[['PassengerId']]
y = df['Age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

dt_regressor = DecisionTreeRegressor(max_depth=5,min_samples_split=10,min_samples_leaf=5,random_state=42)

dt_regressor.fit(X_train, y_train)

y_pred = dt_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE: ", mse)
print("MAE: ", mae)

X1 = df[['PassengerId']]
y1 = df['VIP']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.4, random_state=42)

dt_classifier = DecisionTreeClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,random_state=42)

dt_classifier.fit(X1_train, y1_train)

y1_proba = dt_classifier.predict_proba(X1_test)

fpr, tpr, thresholds = roc_curve(y1_test, y1_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='o')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1])
plt.ylim([0.0, 1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.grid(True)
plt.show()