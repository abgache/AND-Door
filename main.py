import tensorflow as tf
import numpy as np

# Dataset
data_inputs = np.array([[1,1], [0,0], [1,0], [0,1]], dtype=np.int8)
data_outputs = np.array([1, 1, 0, 0], dtype=np.int8)

# Modele
try:
    model.load("model.keras")
except:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraînement
model.fit(data_inputs, data_outputs, epochs=1000, verbose=0)

# Test
preds = model.predict(data_inputs)
print("Predictions :", (preds > 0.5).astype(int).flatten())
model.summary()
model.save("model.keras")