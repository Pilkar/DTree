import time
import pandas as pd
import numpy as np
from sklearn import tree

from DTree import *

def getYX(data):
    d = np.array(data).astype(np.float32)

    Y = d[:, 0]
    X = d[:, 1:]

    return Y, X

SCALE = 5
DEPTH = 12
SIZE = 10000

if __name__ == "__main__":
    data = pd.read_csv('dataset/mnist_train.csv', header=None, nrows=SIZE)
    train_data = data.values.tolist()
    for t in train_data:
        scale_data(t, SCALE)

    data = pd.read_csv('dataset/mnist_test.csv', header=None, nrows=SIZE)
    test_data = data.values.tolist()
    for t in test_data:
        scale_data(t, SCALE)

    print("Our Tree")
    print("Training...")
    start = time.time()

    dtree = DTree(train_data, DEPTH)
    dtree.build_tree()

    end = time.time()
    print("Training took " + str((end - start)) + "s")

    print("Testing...")

    start = time.time()
    out = dtree.classify_multiple(test_data)
    end = time.time()

    print("Testing took " + str((end - start)) + "s")

    test_accuracy = np.mean(out == np.array(test_data)[:, 0])
    print(f"Test accuracy: {round(test_accuracy, 2)}")

    print("Testing...")
    start = time.time()
    out = dtree.classify_multiple(train_data)
    end = time.time()

    print("Testing took " + str((end - start)) + "s")
    train_accuracy = np.mean(out == np.array(train_data)[:, 0])
    print(f"Train accuracy: {round(train_accuracy, 2)}")

    Y, X = getYX(train_data)
    print("Sklearn Tree")
    print("Training...")
    clf = tree.DecisionTreeClassifier(max_depth=DEPTH)
    clf.fit(X, Y)
    ytrain_pred = clf.predict(X)
    print(f"Train accuracy: {round(np.mean(ytrain_pred == Y), 2)}")

    testY, testX = getYX(test_data)

    print("Testing...")
    x = clf.predict(testX)
    print(f"Test accuracy: {round(np.mean(x == testY), 2)}")
