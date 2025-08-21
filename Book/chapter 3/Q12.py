import sys 
import numpy as np

from sklearn.datasets import load_iris 

iris = load_iris(as_frame=True)
print(iris.data.head(3))

X = iris.data[["petal length (cm)",  "petal width (cm)"]].values
print(f"X.shape : {X.shape}")
y = iris["target"].values

X_with_bias = np.c_[np.ones(len(X)), X]
print(X_with_bias)
print(f"X_with_bias.shape : {X_with_bias.shape}")

test_ratio = 0.2 
validation_ratio = 0.2 
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - (test_size + validation_size)

np.random.seed(42)
rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size: -test_size]]
y_valid = y[rnd_indices[train_size: -test_size]]

X_test = X_with_bias[rnd_indices[-test_size : ]]
y_test = y[rnd_indices[-test_size: ]]

def to_one_hot(y):
    return np.diag(np.ones(y.max() + 1))[y]

# print(to_one_hot(y_train[0]))

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)


mean = X_train[:, 1:].mean(axis = 0)
std = X_train[:, 1:].std(axis = 0)

X_train[:, 1:] = (X_train[:, 1:] - mean)/std
X_valid[:, 1:] = (X_valid[:, 1:] - mean)/std
X_test[:, 1:] = (X_test[:, 1:] - mean)/std 

def softmax(logits):
    exps = np.exp(logits)
    exp_sum = exps.sum(axis = 1, keepdims = True)
    return exps/exp_sum

n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))


eta = 0.5
n_epochs = 5001
m = len(X_train)
epsilon = 1e-5

np.random.seed(42)
Theta = np.random.randn(n_inputs, n_outputs)

print(f"Theta.shape : {Theta.shape}")
print(f"theta : {Theta}")

for epoch in range(n_epochs):
    logits = X_train @ Theta
    Y_proba = softmax(logits)
    if epoch % 1000 == 0:
        Y_proba_valid = softmax(X_valid @ Theta)
        xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
        print(epoch, xentropy_losses.sum(axis=1).mean())
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T @ error
    Theta = Theta - eta * gradients

logits = X_valid @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)

accuracy_score = (y_predict == y_valid).mean()
accuracy_score


# I found response 1 to be more accurate and much more
#  close to what the prompt was, though one could argue 
#  that response 2.py was more accurate since it was 
 
#  returning the negative of the number provided, 
#  but that catch is in the prompt itself the prompt 
#  itself explains that the returned number should be in
#   a reverse order so I don't think the user wanted a
#    negative number for this response. There were two 
#    errors in the code I also fixed them, the function 
#    was supposed to get an int as an argument so first I 
#    had to change it to a string so the string reversing 
#    could be done then I changed back the returned value 
#    as an int which was the requirement