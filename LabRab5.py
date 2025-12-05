import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 12))  # 100 примеров, 12 бинарных признаков
Y = np.random.randint(0, 2, size=(100, 2))  # 100 примеров, 2 класса (one-hot encoding)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_single_layer = keras.Sequential([
    keras.layers.Dense(8, activation='sigmoid', input_shape=(12,)),
    keras.layers.Dense(2, activation='softmax')
])

model_single_layer.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_single_layer.fit(X_train, y_train,epochs=100,batch_size=16,validation_data=(X_test, y_test),verbose=1)

test_loss, test_accuracy = model_single_layer.evaluate(X_test, y_test)
print("\nТочность нейросети на тестовых данных: ", test_accuracy)

y_pred_proba = model_single_layer.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

plt.figure(figsize=(6, 4))

plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('График функции ошибки')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("точность на тренировочных данных: ", history.history['accuracy'][-1])
print("точность на тестовых данных: ", test_accuracy)

