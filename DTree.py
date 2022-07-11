import copy
import math
import numpy as np
from Nodes import *


class DTree:

    def __init__(self, data, depth):
        self.root = None
        self.data = data
        self.size = len(data)
        self.attrCount = len(data[0]) - 1
        self.depth = depth
        self.cols_dist = {}
        for i in range(1, len(data[0])):
            self.cols_dist[i] = class_counts(data, i)

    def train(self, data, depth):
        gini, split = self.bestSplit(data)
        if gini == 0 or gini == np.inf or depth == 0:
            return Leaf(data)
        true, false = partition(data, split[0], split[1])
        depth -= 1
        left_branch = self.train(true, copy.deepcopy(depth))
        right_branch = self.train(false, copy.deepcopy(depth))

        return Node(split[0], split[1], left_branch, right_branch)

    def build_tree(self):
        self.root = self.train(self.data, self.depth)

    def predict(self, row, node):
        if isinstance(node, Leaf):
            return node.prediction

        if row[node.attribute] == node.splitValue:
            return self.predict(row, node.left_child)
        else:
            return self.predict(row, node.right_child)

    def classify(self, test):
        out = self.predict(test, self.root)
        return out

    def classify_multiple(self, test):
        out = []
        for row in test:
            out.append(self.predict(row, self.root))
        return out

    def bestSplit(self, data):
        best_gini = np.inf
        best_split = None
        attrs = len(data[0]) - 1

        for f in range(1, attrs):
            if len(self.cols_dist[f]) == 1:
                continue
            values = set([row[f] for row in data])
            for val in values:
                true, false = partition(data, f, val)
                if len(true) == 0 or len(false) == 0:
                    continue
                gt = gini_impurity(true, 0)
                gf = gini_impurity(false, 0)
                value = gt*(len(true)/self.size)+gf*(len(false)/self.size)
                if value < best_gini:
                    best_gini = value
                    best_split = (f, val)
        return best_gini, best_split


def partition(data, col, value):
    true, false = [], []
    for row in data:
        if row[col] == value:
            true.append(row)
        else:
            false.append(row)
    return true, false


def gini_impurity(data, col):
    counts = class_counts(data, col)
    impurity = 1
    for v in counts:
        prob = counts[v] / float(len(data))
        impurity -= prob ** 2
    return impurity


def unique(rows, col):
    return set([row[col] for row in rows])


def scale_data(data, n):
    jump = math.ceil(255/n)
    value = 1/n
    x = 0
    for i in range(1, len(data)):
        for j in range(1, n+1):
            if int(data[i]) <= jump*j:
                data[i] = round(value*(j-1), 2)
                break


