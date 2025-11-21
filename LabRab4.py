import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Загрузка вашего датасета
df = pd.read_csv('processed_titanic.csv')
df2 = pd.read_csv('processed_apple.csv')

X = df[['Age']]
y = df['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(oob_score=True,random_state=42)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("OOB Score: ",rf_model.oob_score_)
print("Test Accuracy: ",accuracy_score(y_test, y_pred_rf))
print("Test ROC-AUC: " ,roc_auc_score(y_test, y_pred_proba_rf))
print(classification_report(y_test, y_pred_rf))