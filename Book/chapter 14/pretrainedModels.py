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

from sklearn.datasets import load_sample_images

model = tf.keras.applications.ResNet50(weights = "imagenet")

images = tf.constant(load_sample_images()["images"])
resizedImages = tf.keras.layers.Resizing(height =224, width = 224, crop_to_aspect_ratio = True)(images)

inputs = tf.keras.applications.resnet50.preprocess_input(resizedImages)

y_prob = model.predict(inputs)

print(f"y_prob.shape : {y_prob.shape}") #should be (2, 1000) as resNet trained on imagenEt classifies 1000 classes and there are 2 images in the photos

topK = tf.keras.applications.resnet50.decode_predictions(y_prob, top = 5)
for imgIndex in range(len(images)):
    print(f"image #{imgIndex}")
    for classId, name, yProb in topK[imgIndex]:
        print(f"classID : {classId}, name : {name}, probability : {yProb: .2%}")

