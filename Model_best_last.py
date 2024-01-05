import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

img_folder = "captchas"
df = pd.read_csv('captchas1/labels.csv', header=None, encoding='utf-8', delimiter=';', names=['text', 'filename'])
characters = sorted(set(''.join(df['text'])))
char_to_num = {char: idx for idx, char in enumerate(characters)}
num_to_char = {str(idx): char for idx, char in enumerate(characters)}
num_to_char['-1'] = ''

max_len = max(df['text'].apply(len))

def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (200, 60))
    label = [char_to_num.get(char, -1) for char in label]
    return img, label

X = [os.path.join(img_folder, filename) for filename in df['filename']]
y = list(df['text'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)  # Drop down test size, maybe it will help with our accuracy

AUTOTUNE = tf.data.AUTOTUNE

def data_generator(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(encode_single_sample, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(16).prefetch(AUTOTUNE)
    return dataset

train_dataset = data_generator(X_train, y_train)
val_dataset = data_generator(X_val, y_val)

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        labels_mask = 1 - tf.cast(tf.equal(y_true, -1), dtype="int64")
        labels_length = tf.reduce_sum(labels_mask, axis=1)
        loss = self.loss_fn(y_true, y_pred, input_length, tf.expand_dims(labels_length, -1))
        self.add_loss(loss)
        return y_pred

def build_model():
    input_img = layers.Input(shape=(200, 60, 3), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="int64")
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    new_shape = ((200 // 4) * (60 // 4) * 64,)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = layers.Dense(len(characters) + 1, activation="softmax")(x)
    output = CTCLayer(name="ctc_loss")(labels, x)
    model = keras.models.Model(inputs=[input_img, labels], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam())
    return model

model = build_model()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
]

history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=callbacks) #Rise test epochs on 50

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(history)
