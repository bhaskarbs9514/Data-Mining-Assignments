import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load & normalize data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Train & evaluate
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=1)
print("Accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])

# Predict custom image
img = cv2.imread('digit1.png', 0)
if img is None:
    print("Image not found"); exit()

img = cv2.resize(img, (28,28))
img = np.invert(img)/255.0
img = img.reshape(1,28,28)

print("Predicted digit:", np.argmax(model.predict(img)))

plt.imshow(img[0], cmap='gray')
plt.axis('off')
plt.show()