import sys 

# print(sys.version_info)
assert(sys.version_info) >= (3, 7) , "version of python below 3.7"

from packaging import version 
import sklearn 
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

print(version.parse(sklearn.__version__))

import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

from pathlib import Path 
# print(Path())
IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents = True, exist_ok = True)

def save_fig(fig_id, tight_layout = True, fig_extension = "png", resolution = 300):
    print(f"in saving figure fig_id : {fig_id}")
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = fig_extension, dpi = resolution)

# from sklearn.datasets import fetch_openml 
# mnist = fetch_openml("mnist_784", as_frame=False)

# X, y = mnist.data, mnist.target
# was having an error to fetch data using fetch_openml so dloading it from tf.keras

from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset from TensorFlow
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Combine train and test data
X = np.vstack((X_train, X_test))  # Combine along rows
y = np.hstack((y_train, y_test))  # Combine along rows

# Reshape X to (70000, 784) and keep y as (70000,)
X = X.reshape(-1, 28 * 28)  # Flatten the 28x28 images into 784 features

# Verify the shapes
print("X shape:", X.shape)  # Should be (70000, 784)
print("y shape:", y.shape)  # Should be (70000,)

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap = "binary")
    plt.axis("off")
some_digit = X[0]
plot_digit(some_digit)
plt.show()

# plotting some images
plt.figure(figsize=(10, 10))
for id, image_data in enumerate(X[:100]):
    plt.subplot(10, 10, id+ 1)
    plot_digit(image_data)
plt.subplots_adjust(wspace=0 ,hspace=0)
save_fig("more digits", tight_layout=False)
plt.show()

# seperating the test case 
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)



from sklearn.linear_model import SGDClassifier 
sgd_clf = SGDClassifier(random_state= 42)
sgd_clf.fit(X_train, y_train_5)

predict5 = sgd_clf.predict([some_digit])
print(predict5)

from sklearn.model_selection import cross_val_score 
print(cross_val_score(sgd_clf,X_train, y_train_5, cv = 3, scoring = "accuracy"))


from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))

print(cross_val_score(dummy_clf, X_train, y_train_5, cv =3, scoring = "accuracy"))

from sklearn.model_selection import cross_val_predict 
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv =3)
print(f"y_train_pred : {y_train_pred}")

from sklearn.metrics import confusion_matrix 
"""
[[TN, FP],
 [FN, TP]]

       Predicted
       Negative  Positive
True
Negative   TN       FP
Positive   FN       TP

"""

cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

# what a perfect confusion matrix should look like
y_train_perfect_predictions = y_train_5
print("perfect confusion")
print(confusion_matrix(y_train_5, y_train_perfect_predictions)) 

from sklearn.metrics import precision_score , recall_score 
print(f"precision_score : {precision_score(y_train_5, y_train_pred)}")
print(f"recall score : {recall_score(y_train_5, y_train_pred)}")

# harmonic mean page 169 
from sklearn.metrics import f1_score 
print(f"f1_score : {f1_score(y_train_5, y_train_pred)}")

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3, method = "decision_function")

from sklearn.metrics import precision_recall_curve 
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
threshold = 3000

plt.plot(thresholds, precisions[: -1], "b--", label = "Precison", linewidth = 2)
plt.plot(thresholds, recalls[:-1], "g--", label = "Recall", linewidth = 2)
plt.vlines(threshold, 0, 1 , "k", "dotted", label = "threshold")

# extra code – this section just beautifies and saves Figure 3–5
idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
save_fig("precision_recall_vs_threshold_plot")


plt.show()

idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]

y_train_pred_90 = (y_scores >= threshold_for_90_precision)

print(f"precision_score(y_train_5, y_train_pred_90) : {precision_score(y_train_5, y_train_pred_90)}")

recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
print(f"recall score : {recall_at_90_precision}")

# calculate ROC curve true positive against 1 - specifity (specifity = true negative) so 1 - specifity = false +ve
from sklearn.metrics import roc_curve 
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores )

idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
print(f"tpr.shape : {tpr.shape}")
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth =2 ,label = "ROC curve")
plt.plot([0, 1], [0, 1], "k", label = "Random classifier ROC Curve")
plt.plot([fpr_90], [tpr_90], "ko", label = "Threshold for 90% precision")

import matplotlib.patches as patches
# extra code – just beautifies and saves Figure 3–7
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.20, 0.89), (0.07, 0.70),
    connectionstyle="arc3,rad=.4",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)
save_fig("roc_curve_plot")

plt.show()


# area under the curve of roc 
from sklearn.metrics import roc_auc_score 
print(f"roc_auc_score(y_train_5 , y_scores) : {roc_auc_score(y_train_5 , y_scores)}")


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")
print(f"class prob for first 2 using randomforest : {y_probas_forest[:2]}")

y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")

# extra code – just beautifies and saves Figure 3–8
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig("pr_curve_comparison_plot")

plt.show()

y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba ≥ 50%
f1_score(y_train_5, y_train_pred_forest)

roc_auc_score(y_train_5, y_scores_forest)

precision_score(y_train_5, y_train_pred_forest) 

recall_score(y_train_5, y_train_pred_forest)

