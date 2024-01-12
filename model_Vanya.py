import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers

img_folder = "captchas"
EPOCHS = 200
df = pd.read_csv('labels.csv', header=None, encoding='utf-8', delimiter=';', names=['text', 'filename'])

data = {row.text: row.filename for row in df.itertuples()}

input = data
characters = sorted(set(''.join(data.keys())))
char_to_num = {v: i for i, v in enumerate(characters)}

num_to_char = {str(i): v for i, v in enumerate(characters)}
num_to_char['-1'] = 'UKN'

print(num_to_char)
from PIL import Image
import glob

files = glob.glob("captchas/*.png")

for file in files:
    print(file)
    img = Image.open(file)
    resize_img = img.resize((200, 60))
    resize_img.save(file)
max_len = len(max(os.listdir("captchas"), key=len)) - 4


def compute_perf_metric(predictions, groundtruth):
    if predictions.shape == groundtruth.shape:
        return np.sum(predictions == groundtruth) / (predictions.shape[0] * predictions.shape[1])
    else:
        raise ValueError()


def encode_single_sample(filename):
    img_path = os.path.join(img_folder, filename)
    img = tf.io.read_file(img_path)

    try:
        img = tf.io.decode_png(img, channels=3)
    except Exception as e:
        print(img_path)
        raise e

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.transpose(img, perm=[1, 0, 2])

    return img.numpy()


def create_train_and_validation_datasets():
    X, y = [], []

    items = list(input.items())
    print(items)
    train_dataset = items
    test_dataset = items[12000:-1]

    y, X = zip(*train_dataset)

    y = list(y)
    y.sort(key=len)
    print(y)
    X = np.asarray(list(map(encode_single_sample, X)))
    y = [list(map(lambda x: char_to_num[x], label)) for label in y]
    y = tf.keras.preprocessing.sequence.pad_sequences(y, max_len, padding='post', value=-1)
    y = np.asarray(y)

    print(X.shape)
    print(y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
    return X_train, X_val, y_train, y_val, test_dataset


X_train, X_val, y_train, y_val, test_dataset = create_train_and_validation_datasets()

print(X_train.shape, X_val.shape)
fig = plt.figure(figsize=(20, 10))
fig.add_subplot(2, 4, 1)
plt.imshow(X_train[0], cmap='gray')
plt.title('Image from X_train with label ' + str(y_train[0]))
plt.axis('off')
fig.add_subplot(2, 4, 2)
plt.imshow(X_train[135], cmap='gray')
plt.title('Image from X_train with label ' + str(y_train[135]))
plt.axis('off')
fig.add_subplot(2, 4, 3)
plt.imshow(X_val[0], cmap='gray')
plt.title('Image from X_val with label ' + str(y_val[0]))
plt.axis('off')
fig.add_subplot(2, 4, 4)
plt.imshow(X_val[23], cmap='gray')
plt.title('Image from X_val with label ' + str(y_val[23]))
plt.axis('off')


def resize_and_normalize(img):
    img = tf.image.resize(img, (200, 60))
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def encode_single_sample(filename):
    img_path = os.path.join(img_folder, filename)
    img = tf.io.read_file(img_path)

    try:
        img = tf.io.decode_png(img, channels=3)
    except Exception as e:
        print(img_path)
        raise e

    img = resize_and_normalize(img)

    return img.numpy()


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
    labels = layers.Input(name="label", shape=(max_len,), dtype="float32")

    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(
        input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Reshape(target_shape=(50, 960), name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = layers.Dense(len(characters) + 1, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_cnn_lstm_model")

    model.compile(optimizer=keras.optimizers.Adam())
    return model


model = build_model()
model.summary()
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(lambda x, y: {'image': resize_and_normalize(x), 'label': y}).batch(16).prefetch(
    buffer_size=tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation_dataset = validation_dataset.map(lambda x, y: {'image': resize_and_normalize(x), 'label': y}).batch(
    16).prefetch(buffer_size=tf.data.AUTOTUNE)
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS,
                    callbacks=[early_stopping, model_checkpoint])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('CTC loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()
prediction_model.save("model.h5")


def get_model(path):
    model = keras.models.load_model(path)
    return model


m = get_model("model.h5")
y_pred = m.predict(X_val)
y_pred = keras.backend.ctc_decode(y_pred, input_length=np.ones(X_val.shape[0]) * 50, greedy=True)
y_pred = y_pred[0][0][0:X_val.shape[0], 0:max_len].numpy()
nrow = 1
fig = plt.figure(figsize=(20, 5))
for i in range(0, 10):
    if i > 4: nrow = 2
    fig.add_subplot(nrow, 5, i + 1)
    plt.imshow(X_val[i].transpose((1, 0, 2)), cmap='gray')
    pred_txt = ''.join(list(map(lambda x: num_to_char[str(x)] if x > -1 else '', y_pred[i])))
    plt.title('Prediction : ' + pred_txt)
    plt.axis('off')
plt.show()
compute_perf_metric(y_pred, y_val)