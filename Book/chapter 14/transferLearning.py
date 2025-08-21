import sys 
assert sys.version_info >= (3, 7)

import tensorflow as tf 
import sklearn 
from packaging import version

assert version.parse(tf.__version__) >= version.parse("2.8.0")
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import matplotlib.pyplot as plt 

plt.rc("font", size = 14)
plt.rc("axes", labelsize = 12, titlesize = 16)
plt.rc("legend", fontsize = 14)
plt.rc("xtick", labelsize = 10)
plt.rc("ytick", labelsize = 10)

import tensorflow_datasets as tfds 
datasets, info = tfds.load("tf_flowers", as_supervised = True, with_info = True)
datasetSize = info.splits["train"].num_examples
print(f"dataset_size : {datasetSize}")
classNames = info.features["label"].names 
numClasses = info.features["label"].num_classes

testSetRaw, validSetRaw, trainSetRaw = tfds.load(
    "tf_flowers", 
    split = ["train[:10%]", "train[10%:25%]", "train[25%:]"], 
    as_supervised=True
)

plt.figure(figsize = (12, 10))
index = 0 
for image, label in validSetRaw.take(9):
    index += 1 
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title(f"Class : {classNames[label]}")
    plt.axis("off")
plt.show()

tf.keras.backend.clear_session()
batchSize = 32 
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height = 224, width = 224, crop_to_aspect_ratio = True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])

trainSet = trainSetRaw.map(lambda X, y : (preprocess(X), y))
trainSet = trainSet.shuffle(1000, seed= 42).batch(batchSize).prefetch(1)

validSet = validSetRaw.map(lambda X, y: (preprocess(X), y)).batch(batchSize)
testSet = trainSetRaw.map(lambda X, y: (preprocess(X), y)).batch(batchSize)

plt.figure(figsize=(12 , 12))
for xBatch, yBatch in validSet.take(1):
    for i in range(9):
        plt.subplot(3, 3, i +1)
        plt.imshow((xBatch[i] + 1)/ 2)
        """
        X_batch[index] + 1: This shifts the range of values. If a pixel value was -1,
         adding 1 makes it 0. If a pixel value was 1, 
         adding 1 makes it 2. So, the range [-1, 1] is shifted to [0, 2].
        """
        plt.title(f"class: {classNames[yBatch[i]]}")
        plt.axis("off")

plt.show()

# augmenting the images, like rotating a bit, flipping them and tweaking contrast
dataAugmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode = "horizontal", seed = 42),
    tf.keras.layers.RandomRotation(factor = 0.1, seed = 42),
    tf.keras.layers.RandomContrast(factor = 0.2, seed = 42)
])

# displaying the same images now that are augmented 
import numpy as np
plt.figure(figsize = (12 , 12))
for xBatch, yBatch in validSet.take(1):
    xBatchAugmented = dataAugmentation(xBatch, training = True)
    plt.title(f"Augmented Images")
    plt.axis("off")
    for i in range(9):
        
        plt.subplot(3, 3, i + 1)
        plt.imshow(np.clip((xBatchAugmented[i] + 1)/2, 0, 1))
        plt.title(f"class : {classNames[yBatch[i]]}")
        plt.axis("off")
plt.show()

# loading the Xception model with the top removed 
tf.random.set_seed(42)
base_model = tf.keras.applications.Xception(weights = "imagenet", include_top= False)

# adding a globalaveragelayer 
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(numClasses, activation = "softmax")(avg)

model = tf.keras.Model(inputs = base_model.inputs, outputs = output)

# freezing the weights of already trained xception model layer
for layer in base_model.layers:
    layer.trainable = False 

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1, momentum = 0.9)
model.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
history = model.fit(trainSet,validation_data = validSet, epochs = 3)

for indices in zip(range(33), range(33, 66), range(66, 99), range(99, 132)):
    for idx in indices:
        print(f"{idx:3}: {base_model.layers[idx].name:22}", end="")
    print()

# unfreezing the base_model layers 56 and above 
for layer in base_model.layers[56:]:
    layer.trainable = True 

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9)
model.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
history = model.fit(trainSet, validation_data = validSet, epochs = 10)

