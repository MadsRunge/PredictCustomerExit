import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score


dataFrame = pd.read_csv("customer_staying_or_not.csv")


print(dataFrame.isnull().sum())


dataFrame.dropna(inplace=True)


print(dataFrame.head())


x = dataFrame.iloc[:, 3:13]
y = dataFrame.iloc[:, -1]


x = pd.get_dummies(x)


columnNames = list(x.columns)


x = x.values
y = y.values


scaler = StandardScaler()
x = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)


model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
adam = Adam(learning_rate=0.001)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=20, verbose=1)


plt.figure(figsize=(10, 6))
sns.lineplot(x=range(len(history.history['loss'])), y=history.history['loss'])
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


model.save('customer_churn_model.keras')
print("Model saved as 'customer_churn_model.keras'")


import pickle
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("Scaler saved as 'scaler.pkl'")


newCustomer = [[600, 40, 3, 60000, 2, 1, 1, 50000, 1, 0, 0, 0, 1]]
newCustomer = scaler.transform(newCustomer)
prediction = model.predict(newCustomer)
print(f"Prediction for new customer: {prediction[0][0]}")
print(f"Likelihood of customer exit: {prediction[0][0] * 100:.2f}%")

