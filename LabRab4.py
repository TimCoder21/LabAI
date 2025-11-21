import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt


# Загрузка вашего датасета
df = pd.read_csv('processed_titanic.csv')
df2 = pd.read_csv('processed_apple.csv')

X = df[['Age']]
y = df['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(oob_score=True,random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)
print("Random Forest")
print("OOB Score: ",rf_model.oob_score_)
print("Test Accuracy: ",accuracy_score(y_test, y_pred_rf), '\n')

ada_model = AdaBoostClassifier(random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
y_pred_proba_ada = ada_model.predict_proba(X_test)[:, 1]
print("AdaBoost")
print("Test Accuracy: ", accuracy_score(y_test, y_pred_ada), '\n')

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]
print("Gradient Boosting")
print("Test Accuracy: ", accuracy_score(y_test, y_pred_gb))


