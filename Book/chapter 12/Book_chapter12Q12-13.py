"""
code for Q12 and Q13 which mainly focuses on custom training loops for training using mnist_fashion dataset
"""


import sys
assert sys.version_info >= (3, 7)

from packaging import version
import tensorflow as tf

assert version.parse(tf.__version__) >= version.parse("2.8.0")


class LayerNormalization(tf.keras.layers.Layer):
  def __init__(self , epsilon = 0.001, **kwargs):
    super().__init__(**kwargs)
    self.epsilon = epsilon
    
  
  def build(self, batch_input_shape):
    self.alpha = self.add_weight(
        name = "alpha",
        shape = batch_input_shape[-1:],
        initializer = tf.keras.initializers.Ones(), # can also write just "ones"
        dtype = self.dtype
    )
    
    self.beta = self.add_weight(
        name = "beta",
        shape = batch_input_shape[-1:],
        initializer = tf.keras.initializers.Zeros(), # can also write "zeros"
        dtype = tf.float32 
    )

  def call(self, X):
    # note : variance = (standard deviation)**2 and symbol for variance is sigma squared 
    mean , variance = tf.nn.moments(X, axes = -1 , keepdims = True)
    std = tf.math.sqrt(variance)
    return self.alpha *  (X - mean)/(std + self.epsilon) + self.beta 


# checking if it produces the same output as tf.keras.layers.LayerNormalization.

x = tf.Variable([[1, 2, 3], [3,4,5]] ,dtype = tf.float32)
x.shape

tf.rank(x)

output = tf.keras.layers.LayerNormalization()(x)
print(f"output from tf : {output}")

output = LayerNormalization()(x)
print(f"output from my Layer : {output}")


# Q13 Train model using custom training loop to tackle the Fashion MNIST

(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()
xTrain.shape

# xTrain = tf.cast(xTrain, tf.float32)/255
# xTest = tf.cast(xTest, tf.float32)/255 

xTrain = xTrain/255.0
xTest = xTest/255.0

import matplotlib.pyplot as plt 
plt.axis('off')
plt.imshow(xTrain[0])

tf.keras.utils.set_random_seed(42)
model = tf.keras.Sequential([ 
    tf.keras.layers.Flatten(input_shape = [28 , 28]),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(300, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "softmax")
])

model.compile(optimizer = "SGD", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(xTrain, yTrain, epochs = 20)

loss, accuracy = model.evaluate(xTest, yTest)
print(f"loss : {loss}, accuracy : {accuracy}")


import numpy as np 
def get_batch(X, y, size = 32):
  idx = np.random.randint(len(X), size = size )
  return X[idx], y[idx]


tf.keras.utils.set_random_seed(42)
model2 = tf.keras.Sequential([ 
    tf.keras.layers.Flatten(input_shape = [28 , 28]),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(300, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "softmax")
])

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
loss_fn = tf.keras.losses.sparse_categorical_crossentropy


epochs = 20
batch_size = 32
for epoch in range(epochs):
  print(f"epoch {epoch + 1}/{epochs}")
  for iteration in range(len(xTrain)// batch_size):
    xBatch, yBatch = get_batch(xTrain, yTrain, batch_size)
    with tf.GradientTape() as tape:
      yPred = model2(xBatch)
      loss = loss_fn(yBatch , yPred)

    gradients = tape.gradient(loss, model2.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model2.trainable_variables))




yPred = model2(xTest)
print(f"yPred.shape: {yPred.shape}")
test_loss = tf.reduce_mean(loss_fn(yTest, yPred)).numpy()
# since yPred is probabiliteis over 10 classes 
predictedLabel = tf.argmax(yPred, axis = 1)
print(f"predictedLabel.dtype : {predictedLabel.dtype}")

accuracy = tf.reduce_mean(tf.cast(tf.equal(predictedLabel, tf.cast(yTest, tf.int64)), tf.float32)).numpy()

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {accuracy}")


# trying the solution stated by gpt
yPred = model2(tf.convert_to_tensor(xTest, dtype=tf.float32))
test_loss = tf.reduce_mean(loss_fn(yTest, yPred)).numpy()
test_accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(yPred, axis=1), yTest), tf.float32)
).numpy()

print(f"loss : {test_loss:.4f}, accuracy : {test_accuracy:.4f}")


# Now trying the solution provided by the book
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test.astype(np.float32) / 255.


model3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(300, activation = "relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])


n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = tf.keras.losses.sparse_categorical_crossentropy
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]


from tqdm.notebook import trange
from collections import OrderedDict

with trange(1, n_epochs + 1, desc="All epochs") as epochs:
  for epoch in epochs:
    with trange(1 , n_steps + 1, desc = f"Epoch {epoch}/{n_epochs}") as steps:
      for step in steps:
        X_batch, y_batch = get_batch(X_train, y_train)
        with tf.GradientTape() as tape:
          y_pred = model3(X_batch)
          main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
          loss = tf.add_n([main_loss] + model3.losses)
          # print(f"loss.shape : {loss.shape}")
        gradients = tape.gradient(loss, model3.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model3.trainable_variables))
        for variable in model3.variables:
          if variable.constraint is not None:
            variable.assign(variable.constraint(variable))

        status = OrderedDict()
        mean_loss(loss)
        status["loss"] = mean_loss.result().numpy()

        for metric in metrics:
          metric(y_batch, y_pred)
          status[metric.name] = metric.result().numpy()
        steps.set_postfix(status)
      
      y_pred = model3(X_valid)
      status["val_loss"] = np.mean(loss_fn(y_valid, y_pred))
      status["val_accuracy"] = np.mean(tf.keras.metrics.sparse_categorical_accuracy(
          tf.constant(y_valid, dtype=np.float32), y_pred))
      steps.set_postfix(status)
    for metric in [mean_loss] + metrics:
      metric.reset_state()
    # test = [mean_loss] + metrics
    # print(test)

ypred = model3(X_test)
predicted_label = tf.argmax(ypred, axis = 1)
testLoss = tf.reduce_mean(loss_fn(y_test, ypred)).numpy()

predicted_label.dtype

acc = tf.reduce_mean(tf.cast(tf.equal(predicted_label, tf.cast(y_test, tf.int64)), tf.float32)).numpy()
acc

