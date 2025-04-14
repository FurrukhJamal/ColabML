import sys 
assert sys.version_info >= (3, 7)

import tensorflow as tf 
import sklearn 
from packaging import version 

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
assert version.parse(tf.__version__) >= version.parse("2.8.0")

(X_train, Y_train), (X_test, Y_text) = tf.keras.datasets.cifar10.load_data()

classNames = ["airplane", "automobile" , "bird" , "cat", "deer", "dog", "frog" , "horse", "ship", "truck"]

import matplotlib.pyplot as plt 
plt.figure(figsize=(3,3))
plt.axis("off")
plt.imshow(X_train[0])
plt.title(classNames[int(Y_train[0])])
plt.show()

from sklearn.model_selection import train_test_split
xTrain,  xVal, yTrain, yVal = train_test_split(X_train, Y_train, test_size= 0.2, random_state= 42)

print(f"xTrain.shape : {xTrain.shape}")
print(f"xVal.shape : {xVal.shape}")
print(f"X_test.shape : {X_test.shape}")

from pathlib import Path 
from time import strftime 

def get_run_logdir(root_logdir = "my_cifar10_logs"):
    return Path(root_logdir)/f"run_{strftime("%Y_%m_%d_%H_%M_%S")}"

runLogDir = get_run_logdir()
# callbacks
tensorboard_cb = tf.keras.callbacks.TensorBoard(runLogDir)
earlyStopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)
callbacks = [earlyStopping_cb, tensorboard_cb]

tf.random.set_seed(42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (32, 32, 3)))

for _ in range(20):
    model.add(tf.keras.layers.Dense(100, activation = "swish", kernel_initializer = "he_normal"))

# adding the output layer
model.add(tf.keras.layers.Dense(10, activation = "softmax"))

optimizer = tf.keras.optimizers.Nadam(learning_rate = 3e-5)

# model.compile(loss = "sparse_categorical_crossentropy", 
#               optimizer = optimizer,
#               metrics = ["accuracy"])

# model.fit(xTrain, yTrain, epochs = 100, validation_data = (xVal, yVal), callbacks = callbacks)
# loss , accuracy = model.evaluate(X_test, Y_text)

# print(f"loss : {loss} accuracy : {accuracy}")

# part C: adding batch normalization 

# tf.random.set_seed(42)

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape = (32,32,3)))
# for _ in range(20):
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.Dense(100, activation = "swish", kernel_initializer = "he_normal"))

# model.add(tf.keras.layers.Dense(10, activation = "softmax"))

# optimizer = tf.keras.optimizers.Nadam(learning_rate = 5e-4)

# model.compile(loss = "sparse_categorical_crossentropy", 
#               optimizer = optimizer,
#               metrics = ["accuracy"])

# runLogDir = get_run_logdir("my_cifar10_bn_model")

# tensorboard_cb = tf.keras.callbacks.TensorBoard(runLogDir)
# earlyStopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)
# callbacks = [earlyStopping_cb, tensorboard_cb]

# model.fit(xTrain, yTrain, epochs = 100, validation_data = (xVal, yVal), callbacks = callbacks)
# loss , accuracy = model.evaluate(xVal, yVal)

# print(f"loss : {loss} accuracy : {accuracy}")

# part D: using Selu

tf.random.set_seed(42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(tf.keras.layers.Dense(100,
                                    kernel_initializer="lecun_normal",
                                    activation="selu"))

model.add(tf.keras.layers.Dense(10, activation="softmax"))

optimizer = tf.keras.optimizers.Nadam(learning_rate=7e-4)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=20, restore_best_weights=True)


run_logdir = get_run_logdir("my_cifar10_selu")
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb,  tensorboard_cb]

X_means = xTrain.mean(axis=0)
X_stds = xTrain.std(axis=0)
X_train_scaled = (xTrain - X_means) / X_stds
X_valid_scaled = (xVal - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

model.fit(X_train_scaled, yTrain, epochs=100,
          validation_data=(X_valid_scaled, yVal),
          callbacks=callbacks)

model.evaluate(X_valid_scaled, yVal)