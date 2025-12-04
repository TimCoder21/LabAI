import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, RocCurveDisplay, roc_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt


from ucimlrepo import fetch_ucirepo
spambase = fetch_ucirepo(id=94)
# data (as pandas dataframes)
X = spambase.data.features
y = spambase.data.targets
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(oob_score=True,random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest")
print("OOB Score: ",rf_model.oob_score_)
print("Test Accuracy: ",accuracy_score(y_test, y_pred_rf), '\n')

ada_model = AdaBoostClassifier(random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
print("AdaBoost")
print("Test Accuracy: ", accuracy_score(y_test, y_pred_ada), '\n')

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting")
print("Test Accuracy: ", accuracy_score(y_test, y_pred_gb))

fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax, name='Random Forest')
RocCurveDisplay.from_estimator(ada_model, X_test, y_test, ax=ax, name='AdaBoost')
RocCurveDisplay.from_estimator(gb_model, X_test, y_test, ax=ax, name='Gradient Boosting')
plt.ylabel('TPR')
plt.xlabel('FPR')

ax.plot([0,1], [0, 1], linestyle='--', lw=2, color='r')

ax.set_title('ROC-кривые')
plt.show()

