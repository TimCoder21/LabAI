import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("processed_titanic.csv")
df2 = pd.read_csv("processed_apple.csv")

X = df2[['Date']]
y = df2['Adj Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

linear_model = LinearRegression()

from sklearn.preprocessing import PolynomialFeatures
n = 10
poly_features = PolynomialFeatures(n)
X_train = poly_features.fit_transform(X_train)
linear_model.fit(X_train, y_train)

X_test = poly_features.transform(X_test)

y_pred_test = linear_model.predict(X_test)

print("Оценка регресии")
MSE = mean_squared_error(y_test, y_pred_test)
print("MSE: ", MSE)
RMSE = root_mean_squared_error(y_test, y_pred_test)
print("RMSE: ", RMSE)
MAE = mean_absolute_error(y_test, y_pred_test)
print("MAE: ", MAE)

from ucimlrepo import fetch_ucirepo

# fetch dataset
spambase = fetch_ucirepo(id=94)

# data (as pandas dataframes)
X1 = spambase.data.features
y1 = spambase.data.targets

# metadata
print(spambase.metadata)

# variable information
print(spambase.variables)




#X1 = df[['Age']]
#y1 = df['Transported']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.4, random_state=42)
X1_test, X1_val, y1_test, y1_val = train_test_split(X1_test, y1_test, test_size=0.4, random_state=42)


logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X1_train, y1_train)
y1_pred_test = logreg_model.predict(X1_test)

accuracy = accuracy_score(y1_test, y1_pred_test)
print("Точность: " , accuracy)



cm = confusion_matrix(y1_test, y1_pred_test)
print(cm)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()



