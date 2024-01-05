import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

# Загрузка данных
img_folder = "captchas"
df = pd.read_csv('labels.csv', header=None, encoding='utf-8', delimiter=';', names=['text', 'filename'])
characters = sorted(set(char for label in df['text'] for char in label))
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
num_to_char = {idx + 1: char for idx, char in enumerate(characters)}
max_len = max(len(s) for s in df['text'])

# Предобработка данных
def encode_single_sample(img_path, label):
    # Изображение
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (200, 60)) # Надо уточнить про точный размер, всегда ли 200 на 60 или можно увеличть -> Точноть выше
    # Метка
    label = [char_to_num[c] for c in label]
    return img, label

# Создание набора данных
def create_dataset(df):
    img_paths = [os.path.join(img_folder, filename) for filename in df['filename']]
    labels = list(df['text'])
    X, y = zip(*(encode_single_sample(path, label) for path, label in zip(img_paths, labels)))
    return np.array(X), np.array(y)

# Разделение данных
X, y = create_dataset(df)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42) #Заменил тестовый размер

# Определение модели CRNN с CTC loss в TensorFlow
class CRNNCTCModel(tf.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")
        self.mp1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.mp2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.reshape = tf.keras.layers.Reshape((-1, 64 * 15))
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.blstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))
        self.blstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
        self.dense2 = tf.keras.layers.Dense(num_classes)

    def __call__(self, x, training=False):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.reshape(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.blstm1(x)
        x = self.blstm2(x)
        return self.dense2(x)

# Обучение и оптимизация модели
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = tf.reduce_mean(tf.nn.ctc_loss(
            labels, logits, label_length=np.full([labels.shape[0]], max_len),
            logit_length=np.full([logits.shape[0]], logits.shape[1]),
            logits_time_major=False, blank_index=0))
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

# Точность предсказаний
def accuracy(logits, labels):
    decoded_labels = tf.keras.backend.ctc_decode(logits, input_length=np.full([logits.shape[0]], logits.shape[1]), greedy=True)
    return np.mean([np.array_equal(a, b) for a, b in zip(decoded_labels[0][0].numpy(), labels)])

# Создание экземпляра модели и оптимизатора
model = CRNNCTCModel(num_classes=len(characters) + 1)
optimizer = tf.optimizers.Adam()

# Цикл обучения
for epoch in range(10):  # Примерное количество эпох, может потребоваться корректировка
    for batch_idx in range(0, len(X_train), 32):  # Размер партии для обучения
        batch_images = X_train[batch_idx:batch_idx+32]
        batch_labels = y_train[batch_idx:batch_idx+32]
        loss = train_step(model, batch_images, batch_labels)
    val_logits = model(X_val, training=False)
    val_accuracy = accuracy(val_logits, y_val)
    print(f"Epoch {epoch}: Loss: {loss.numpy()}, Validation Accuracy: {val_accuracy}")

# нужно больше проверок точности и потерь, токенизацию надо поменять просто на посимвольное разбиение капчи, оно идет как одно слово, просто символы в токенизацию с точным и неизменным ID

def meteor_metric(text, text_sum):
    if isinstance(text_sum, str):
        return round(meteor([word_tokenize(text)], word_tokenize(text_sum)), 4)
    else:
        return 0


def bleu_metric(reference, hypothesis):
    reference = [word_tokenize(reference)]
    hypothesis = word_tokenize(hypothesis)
    return round(sentence_bleu(reference, hypothesis), 4)


def nist_metric(reference, hypothesis):
    try:
        reference = [word_tokenize(reference)]
        hypothesis = word_tokenize(hypothesis)
        return round(sentence_nist(reference, hypothesis), 4)
    except ZeroDivisionError:
        return 0

