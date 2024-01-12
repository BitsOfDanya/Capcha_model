import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, LSTM, Bidirectional, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Загрузка и подготовка данных
print("Загрузка данных...")
df = pd.read_csv('A:/WORK/Programming/DataScience and ML/Capcha (Ваня)/Capcha_model/labels.csv', header=None, delimiter=';', names=['label', 'filename'])
base_path = 'A:/WORK/Programming/DataScience and ML/Capcha (Ваня)/Capcha_model/captchas1/'
df['filename'] = df['filename'].apply(lambda x: os.path.join(base_path, x))
df = df[df['filename'].apply(os.path.exists)]
characters = sorted(set(char for label in df.label for char in label))
char_to_num = {char: i for i, char in enumerate(characters)}
num_to_char = {i: char for char, i in char_to_num.items()}
img_width, img_height = 200, 60
num_classes = len(characters) + 1
max_text_length = max(len(label) for label in df.label)
max_label_value = max(char_to_num.values()) + 1

def encode_label(text):
    return [char_to_num[char] for char in text] + [max_label_value] * (max_text_length - len(text))

def decode_label(nums):
    return ''.join(num_to_char[num] for num in nums if num in num_to_char)

def preprocess_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    return img

paths = df['filename'].values
labels = [encode_label(label) for label in df['label'].values]
paths_train, paths_val, labels_train, labels_val = train_test_split(paths, labels, test_size=0.1, random_state=42)

# Создание модели
print("Создание модели...")
input_img = Input(shape=(img_height, img_width, 3), name='image', dtype='float32')
x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_img)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)  # Добавлен слой Dropout
x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)  # Добавлен ещё один слой Dropout
x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.5)(x)  # Увеличен коэффициент Dropout для этого слоя

new_shape = ((img_width // 16), (img_height // 16) * 256)
x = Reshape(target_shape=new_shape)(x)
x = Dense(128, activation='relu')(x)

new_shape = ((img_width // 16), (img_height // 16) * 256)
x = Reshape(target_shape=new_shape)(x)
x = Dense(128, activation='relu')(x)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

x = TransformerBlock(128, num_heads=4, ff_dim=64)(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)
x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(x)
output = Dense(num_classes, activation='softmax', name='output')(x)

labels_input = Input(name='label', shape=[max_text_length], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = layers.Lambda(lambda args: tf.keras.backend.ctc_batch_cost(*args), name='ctc')([labels_input, output, input_length, label_length])
model = Model(inputs=[input_img, labels_input, input_length, label_length], outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-4))

# Создание наборов данных
def create_dataset(paths, labels, batch_size=32):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    img_ds = path_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    input_length = np.ones((len(paths), 1)) * (img_width // 16 - 2)
    label_length = np.array([[len(label) - np.count_nonzero(label == max_label_value)] for label in labels])
    input_length_ds = tf.data.Dataset.from_tensor_slices(input_length)
    label_length_ds = tf.data.Dataset.from_tensor_slices(label_length)
    ds = tf.data.Dataset.zip((img_ds, label_ds, input_length_ds, label_length_ds))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(paths_train, labels_train)
val_ds = create_dataset(paths_val, labels_val)

# Обучение модели
print("Обучение модели...")
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stopping, model_checkpoint])

# Визуализация истории обучения
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy Over Epochs')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

plot_history(history)

# Декодирование предсказаний
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * (pred.shape[1] - 2)
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    return [''.join(decode_label(res.numpy())) for res in results]

# Визуализация и сохранение результатов
def visualize_and_save_results(image_paths, predictions, labels):
    num_images = len(image_paths)
    if num_images == 0:
        return

    rows = num_images // 2 if num_images > 1 else 1
    cols = 2 if num_images > 1 else num_images

    plt.figure(figsize=(15, 5 * rows))
    with open("predictions2.txt", "w") as file:
        for i in range(num_images):
            plt.subplot(rows, cols, i + 1)
            img = plt.imread(image_paths[i])
            plt.imshow(img)
            title = f'Предсказание: {predictions[i]}\nОригинал: {decode_label(labels[i])}'
            plt.title(title)
            plt.axis('off')
            file.write(f'{os.path.basename(image_paths[i])}: {title}\n')
    plt.tight_layout()
    plt.show()

# Проверка на переобучение
train_loss = history.history['loss']
val_loss = history.history['val_loss']
if val_loss[-1] > train_loss[-1]:
    print("Возможное переобучение!")
else:
    print("Переобучение отсутствует.")