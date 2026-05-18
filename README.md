# 1
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Load Dataset
data = pd.read_csv("housing.csv", delim_whitespace=True, header=None)
# Display Dataset
print("First 5 Rows of Dataset:\n")
print(data.head())
# Display Shape
print("\nDataset Shape:", data.shape)
# Split Features and Target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create Deep Neural Network Model
model = Sequential()
# Input Layer + Hidden Layer 1
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
# Hidden Layer 2
model.add(Dense(32, activation='relu'))
# Hidden Layer 3
model.add(Dense(16, activation='relu'))
# Output Layer
model.add(Dense(1))
# Compile Model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)
# Display Model Summary
print("\nModel Summary:\n")
model.summary()
# Train the Model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)
# Predict Values
y_pred = model.predict(X_test)
# Calculate Performance Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# Display Results
print("\nModel Performance")
print("-------------------")
print("Mean Squared Error :", mse)
print("Mean Absolute Error:", mae)
# Display Sample Predictions
print("\nSample Predictions:\n")

for i in range(5):
    print("Actual Price =", y_test[i],
          " Predicted Price =", y_pred[i][0])
# 2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
# Load dataset
df = pd.read_csv("IMDB_Dataset.csv",
                 encoding='latin1',
                 engine='python',
                 on_bad_lines='skip')
# Convert labels
encoder = LabelEncoder()
df['sentiment'] = encoder.fit_transform(df['sentiment'])
# Input and output
X = df['review']
y = df['sentiment']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# Padding
X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)
# Build model
model = Sequential()

model.add(Embedding(5000, 64, input_length=100))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Train model
model.fit(X_train, y_train,
          epochs=3,
          batch_size=64)
# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

# 3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)
sns.set_style('whitegrid')


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('Training Data Shape:', x_train.shape)
print('Testing Data Shape :', x_test.shape)


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')

plt.show()


model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

print('Test Accuracy:', test_accuracy)
print('Test Loss:', test_loss)
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Fashion MNIST Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

