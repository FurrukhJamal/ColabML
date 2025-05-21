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

# print(sys.modules.keys())
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from sklearn.datasets import load_sample_images 
images = load_sample_images()["images"]
print(f"len(images) : {len(images)}")
print(f"images.shape : {images[0].shape}")

plt.subplot( 1, 2 , 1)
plt.axis("off")
plt.imshow(images[0])
plt.subplot(1, 2, 2)
plt.imshow(images[1])
plt.axis("off")
plt.show()

images = tf.keras.layers.CenterCrop(height = 70, width =120)(images)
images = tf.keras.layers.Rescaling(scale = 1/255)(images)

plt.title("images after CentreCrop")
plt.axis("off")
plt.subplot( 1, 2 , 1)
plt.axis("off")
plt.imshow(images[0])
plt.subplot(1, 2, 2)
plt.imshow(images[1])
plt.axis("off")
plt.show()

print(f"images.shape : {images.shape}")

conv_layer = tf.keras.layers.Conv2D(filters = 32, kernel_size=7)
fmaps = conv_layer(images)
print(f"fmaps.shape : {fmaps.shape}") #(TensorShape([2, 64, 114, 32]))

plt.figure(figsize = (15, 9))
for img_id in (0, 1):
    for index, fmap_id in enumerate((0, 1, 3, 29)):
        # print(index)
        plt.subplot(2, 4, img_id* 4 + index + 1)
        plt.imshow(fmaps[img_id, :, :, fmap_id], cmap="gray")
        plt.axis("off")
plt.show()


mnist = tf.keras.datasets.fashion_mnist.load_data()
(xTrainFull, yTrainFull), (xTest, yTest) = mnist
import numpy as np
print(f"xTrainFull.shape before expanding dim : {xTrainFull.shape}")
xTrainFull = np.expand_dims(xTrainFull, axis = -1).astype(np.float32)/255
xTest = np.expand_dims(xTest.astype(np.float32), axis = -1)/255
print(f"xTrainFull.shape after expanding dim : {xTrainFull.shape}")

xTrain, xValid = xTrainFull[:-5000], xTrainFull[-5000:]
yTrain, yValid = yTrainFull[:-5000], yTrainFull[-5000:] 

from functools import partial 
tf.random.set_seed(42)
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size = 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")

model = tf.keras.Sequential([
    DefaultConv2D(filters = 64, kernel_size =7, input_shape = [28, 28 ,1]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters = 128),
    DefaultConv2D(filters = 128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters = 256),
    DefaultConv2D(filters = 256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = "relu", kernel_initializer = "he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation = "relu", kernel_initializer = "he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation = "softmax")
])

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "nadam", metrics = ["accuracy"])
history = model.fit(xTrain, yTrain, epochs = 10, validation_data = (xValid, yValid))

loss, acc = model.evaluate(xTest, yTest)
print(f"accuracy : {acc}, loss : {loss}")
y_pred = np.argmax(model.predict(xTest[1]))

print(f"actual class : {yTest[1]}, predicted : {y_pred}")






