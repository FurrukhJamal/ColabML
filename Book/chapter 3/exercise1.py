import sklearn
import sys 

assert(sys.version_info) >= (3 , 7)
from packaging import version
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 

from scipy.ndimage import shift

def loadData():
    (xTrain , yTrain), (xTest, yTest) = mnist.load_data()
    # print(f"xTrain.shape : {xTrain.shape}")
    xTrain = xTrain.reshape(-1, 28 * 28)
    xTest = xTest.reshape(-1, 28 * 28)
    return (xTrain , yTrain), (xTest, yTest)

def shiftedImage(image, dx , dy):
    # convert the image (28 , 28)
    img = image.reshape((28, 28))
    shifted_image = shift(img, (dy, dx), mode = "constant", cval = 0)
    # print(f"shifted_image.shape : {shifted_image.shape}")
    return shifted_image.reshape([-1])
    
def plot_digit(image_data, title):
    image = image_data.reshape(28, 28)
    plt.title(title)
    plt.imshow(image, cmap = "binary")
    

def shiftImageLeft(image, pixels):
    return shiftedImage(image, -pixels, 0)

def shiftImageRight(image, pixels):
    return shiftedImage(image, pixels, 0)

def shiftImageUp(image, pixels):
    return shiftedImage(image, 0 , -pixels)

def shiftImageDown(image, pixels):
    return shiftedImage(image, 0 , pixels)


if __name__ == "__main__":
    (xTrain , yTrain), (xTest, yTest) = loadData()
    knn_clf = KNeighborsClassifier()
    # knn_clf.fit(xTrain, yTrain)
    from sklearn.model_selection import cross_val_score, GridSearchCV
    
    score = cross_val_score(knn_clf, xTrain, yTrain, cv = 3, scoring = "accuracy")
    print(f"score : {score}")
    print(knn_clf.get_params())

    param_grid = {
        "weights" : ["uniform", "distance"],
        "n_neighbors" : [3, 4, 5]

    }
    gridSearch = GridSearchCV(knn_clf, param_grid, cv =2, n_jobs= 3)
    gridSearch.fit(xTrain, yTrain)

    print(f"best training score : {gridSearch.best_score_}")
    print(f"best_params : {gridSearch.best_params_}")

    testAccuracy = gridSearch.score(xTest, yTest)
    print(f"test accuracy : {testAccuracy}")
    prediction = gridSearch.predict([xTrain[0]])
    print(f"prediction : {prediction}")

    bestEstimator = gridSearch.best_estimator_

    # for Q2
    img = xTrain[0]
    plt.figure(figsize= (10, 10))
    for i, orientation in zip( range(3), ["original", "shifted right", "shifted down"]):
        plt.subplot(1 , 3, i + 1)
        if i == 0:
            plot_digit(img, orientation)
        elif i == 1:
            plot_digit(shiftImageRight(img, 5), orientation)
        else:
            plot_digit(shiftImageDown(img, 5), orientation)
    plt.show()

    # testImages = xTrain[0: 5, :]
    # testY = yTrain[:5]
    # print(f"testImages.shape : {testImages.shape}")
    # print(f"testY.shape : {testY.shape}")
    testImages = xTrain.copy()
    testY = yTrain.copy()
    rows, _ = testImages.shape
    imagesWithAugmentedImages = testImages.copy()
    resultImages = []
    resultY = []
    for i in range(rows):
        print(f"augmenting image number : {i}")
        img =  testImages[i]
        newImg = shiftImageRight(img, 1)
        resultImages.append(newImg)
        resultY.append(testY[i])

        newImg = shiftImageDown(img, 1)
        resultImages.append(newImg)
        resultY.append(testY[i])

        newImg = shiftImageLeft(img, 1)
        resultImages.append(newImg)
        resultY.append(testY[i])

        newImg = shiftImageUp(img, 1)
        resultImages.append(newImg)
        resultY.append(testY[i])
    print("Augmentaion of images finished")
    xNew = np.array(resultImages)
    print(f"xNew.shape : {xNew.shape}")
    yNew = np.array(resultY)
    print(f"yNew.shape : {yNew.shape}")

    # shuffling the data
    shuffleIndices = np.random.permutation(len(xNew))
    shuffledX = xNew[shuffleIndices]
    shuffledY = yNew[shuffleIndices]

    # adding to the original train data
    augmentedX = np.vstack((xTrain, shuffledX[:30000, :]))
    augmentedY = np.hstack((yTrain, shuffledY[:30000]))

    print(f"augmentedX.shape : {augmentedX.shape}")
    print(f"augmentedY.shape : {augmentedY.shape}") 

    # getting the new score with augmented trained data
    gridSearch.fit(augmentedX, augmentedY)
    print(f"best score with augmented data : {gridSearch.best_score_}")

    bestEstimator = gridSearch.best_estimator_
    accuracy = bestEstimator.score(xTest, yTest)
    print(f"accuracy with augmented testing is : {accuracy}")    

