import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 12))  # 100 примеров, 12 бинарных признаков
y = np.random.randint(0, 2, size=(100, 2))  # 100 примеров, 2 класса (one-hot encoding)

# Разделяем на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
   keras.layers.Dense(8, activation='relu', input_shape=(12,)),  # Скрытый слой 1
   keras.layers.Dense(2, activation='sigmoid')                    # Выходной слой (бинарная классификация)
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Оцениваем качество на тестовой выборке
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test, y_pred))

# График изменения функции ошибки
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

fpr, tpr, thresholds = roc_curve(y_test, history)
plt.figure(1, figsize=(8, 6))
plt.plot(fpr, tpr, marker='o')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1])
plt.ylim([0.0, 1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.grid(True)
plt.show()