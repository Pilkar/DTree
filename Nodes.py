
class Node:
    def __init__(self, attr, val, left, right, ):
        self.left_child = left
        self.right_child = right
        self.attribute = attr
        self.splitValue = val


class Leaf:
    def __init__(self, data):
        self.predictions = class_counts(data, 0)
        self.prediction = self.predict(data)

    def predict(self, data):
        if len(self.predictions) == 1:
            return list(self.predictions.keys())[0]
        else:
            best_prob = list(self.predictions.keys())[0]
            prob = self.predictions[best_prob]/len(data)
            tmp = 0
            for k in self.predictions:
                tmp = self.predictions[k]/len(data)
                if tmp > prob:
                    prob = tmp
                    best_prob = k
            return best_prob


def class_counts(rows, col):
    counts = {}
    for row in rows:
        label = row[col]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts
