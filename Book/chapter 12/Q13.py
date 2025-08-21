import sys 
assert sys.version_info >= (3, 7)

from packaging import version 
import tensorflow as tf 
assert version.parse(tf.__version__) >= version.parse("2.8.0")

# load the mnist fashion dataset
(xTrainFull, yTrainFull), (xTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()
xTrainFull = xTrainFull/255.0
xTest = xTest/255.0

print(f"xTrainFull.shape : {xTrainFull.shape}")

xVal = xTrainFull[:5000]
yVal = yTrainFull[:5000]

xTrain = xTrainFull[5000:]
yTrain = yTrainFull[5000:]

n_epochs = 10
batchSize = 32 
n_steps = len(xTrain)//batchSize 
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01) 
loss_fn = tf.keras.losses.sparse_categorical_crossentropy
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = [28 ,28]),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "softmax")
])

import numpy as np
def get_random_batch(X, labels, batchSize = 32):
    idx = np.random.randint(len(X), size = batchSize)
    return X[idx], labels[idx]


from tqdm import trange
from collections import OrderedDict

with trange(1, n_epochs + 1, desc = "All epochs") as epochs:
    for epoch in epochs:
        with trange(1, n_steps + 1, desc = f"{epoch}/{n_epochs}") as steps:
            for step in steps:
                xBatch, yBatch = get_random_batch(xTrain, yTrain)
                with tf.GradientTape() as tape:
                    yPred = model(xBatch)
                    mainLoss = tf.reduce_mean(loss_fn(yBatch, yPred))
                    loss = tf.add_n([mainLoss] + model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                for variable in model.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                status = OrderedDict()
                mean_loss(loss)
                status["loss"] = mean_loss.result().numpy()
                for metric in metrics:
                    metric(yBatch, yPred)
                    status[metric.name] = metric.result().numpy()
                steps.set_postfix(status)
            yValPred = model(xVal)
            status["val loss"] = tf.reduce_mean(loss_fn(yVal, yValPred)).numpy()
            status["val acc"] = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(tf.constant(yVal), yValPred)).numpy()
            epochs.set_postfix(status)
        for metric in [mean_loss] + metrics:
            metric.reset_state()

yPred = model(xTest)
predictedLabel = tf.argmax(yPred, axis = 1)
test_loss = tf.reduce_mean(loss_fn(yTest, yPred)).numpy() 
acc = tf.reduce_mean(tf.cast(tf.equal(predictedLabel, tf.cast(yTest, tf.int64)), tf.float32)).numpy()

print(f"accuracy : {acc} loss : {test_loss}")