"""
Train a deep MLP on the MNIST dataset (you can load it using
tf.keras. data sets.mnist.load_data()). See if you can
get over 98% accuracy by manually tuning the hyperparameters.
"""
import sys
assert sys.version_info >= (3, 7)

from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.0")


import keras_tuner as kt
import tensorflow as tf
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

def model_build(hp):
  n_hidden = hp.Int("n_hidden", min_value = 0, max_value = 4, default = 2)
  n_neurons = hp.Int("n_neurons", min_value = 100, max_value = 300)
  learning_rate = hp.Float("learning_rate", min_value = 1e-4, max_value = 1e-1, sampling = "log")
  optimizer = hp.Choice("optimizer", values = ["sgd", "adam"])
  if optimizer == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)
  else:
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape= [28, 28]))
  for _ in range(n_hidden):
    model.add(tf.keras.layers.Dense(n_neurons, activation = "relu"))
  model.add(tf.keras.layers.Dense(10, activation = "softmax"))

  model.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

  return model


random_search_tuner = kt.RandomSearch(model_build, objective = "val_accuracy", max_trials = 15, overwrite = False, directory = "my first test", project_name = "mnist again", seed = 42)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)

random_search_tuner.search(X_train, Y_train, epochs = 100, validation_data = (X_val, Y_val), callbacks= [early_stopping_cb])

top3_models = random_search_tuner.get_best_models(num_models = 3)
best_model = top3_models[0]


best_hyperparameters = random_search_tuner.get_best_hyperparameters(num_trials = 1)[0]
print(best_hyperparameters.values)

test_loss, test_accuracy = best_model.evaluate(X_test, Y_test)
print(f"test_loss : {test_loss} test_accuracy : {test_accuracy}")