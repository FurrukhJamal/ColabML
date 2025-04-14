import sys 
assert sys.version_info >= (3, 7)

from packaging import version 
import sklearn 
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

from sklearn.datasets import make_moons 
X_moons, y_moons = make_moons(n_samples = 10000, noise = 0.4, random_state = 43)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_moons, y_moons, train_size= 0.2, random_state= 42)

from sklearn.model_selection import GridSearchCV

params = {
    "max_leaf_nodes" : list(range(2, 100)),
    "min_samples_split" : [2, 3, 4],
    "max_depth" : list(range(1, 7))
}

from sklearn.tree import DecisionTreeClassifier

gridSearch  = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=3 , n_jobs=5)
gridSearch.fit(X_train, Y_train)

print(f"best estimator : {gridSearch.best_estimator_}")

from sklearn.metrics import accuracy_score

y_pred = gridSearch.predict(X_test)

print(f"accuracy is : {accuracy_score(Y_test, y_pred)}")


print("\n\n QUESTION 8")
print(f"len(X_train) : {len(X_train)}")

from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances  = 100 

rs = ShuffleSplit(n_splits=n_trees, test_size= len(X_train) - n_instances, random_state=42)

mini_sets = []

for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    Y_mini_train = Y_train[mini_train_index]
    mini_sets.append((X_mini_train, Y_mini_train))

from sklearn.base import clone
forest = [clone(gridSearch.best_estimator_) for _ in range(n_trees)] 

accuracy_scores = []
import numpy as np 
for tree , (X_mini_train, Y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, Y_mini_train)

    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(Y_test, y_pred))

print(f"accuracy score : {np.mean(accuracy_scores)}")

Y_pred = np.empty([n_trees, len(X_test)], dtype = np.uint8)

print(f"Y_pred.shape : {Y_pred.shape}")

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

from scipy.stats import mode
"""
from a shape of 1000, 8000 mode reduces the array to 8000
predicting class based on max number of votes for that class
"""
print(f"Y_pred: {Y_pred}")
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
print(f"y_pred_majority_votes : {y_pred_majority_votes}")
print(f"y_pred_majority_votes.shape : {y_pred_majority_votes.shape}")
print(f"n_votes : {n_votes}")

print(accuracy_score(Y_test, y_pred_majority_votes.reshape([-1])))

