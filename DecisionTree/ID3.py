import sys
import math
import Import
import numpy as np
import random

# Variables
data_type = "bank"
purity_type = "ig"
max_depth = -1


# Structures
class Tree:
    def __init__(self, purity="ig"):
        self.purity = purity
        self.max_depth = max_depth
        self.root = None

    def set_root(self, node):
        self.root = node


class Node:
    def __init__(self, s, _example_weights, parent, is_leaf):
        self.s = s
        self.example_weights = _example_weights
        self.parent = parent
        self.is_leaf = is_leaf
        self.branches = {}
        self.attribute = -1
        self.label = None

    def set_attribute(self, attribute):
        self.attribute = attribute

    def set_label(self, label):
        self.label = label

    def add_branch(self, value, node):
        self.branches[value] = node


# Normal Decision Tree
def _run():
    """Runs the normal ID3 algorithm"""
    tree = train(train_data, 0)
    print("TRAIN: ", calculate_prediction_error_for_tree(train_data, tree.root))
    print("TEST: ", calculate_prediction_error_for_tree(test_data, tree.root))


# Attributes
def _get_attributes(_attributes):
    """
    Gets a subset of attributes for the next attribute split if there is a set subset size.
    Otherwise just returns the unused attributes
    """
    if attr_subset_num != 0:
        return _get_subset_of_attributes(_attributes)
    else:
        return _attributes


def _get_subset_of_attributes(_attributes):
    """Gets a random subset of attributes that haven't been split on"""
    subset = []
    while len(subset) < attr_subset_num and len(subset) < len(_attributes):
        n = random.randint(0, len(attributes) - 1)
        if attributes[n] in _attributes:
            subset.append(attributes[n])
    return subset


# ID3
def train(s, t, _attr_subset_num=0):
    """
    Trains a decision tree using the ID3 algorithm using the type of purity function given.
    :param s: [](examples, features) - the entire dataset
    :param t: the index of example_weights to use. ie the training iteration.
    :param _attr_subset_num: If running Random Forest, set the attribute subset size.
    :return: The trained tree
    """
    global attr_subset_num
    attr_subset_num = _attr_subset_num

    _tree = Tree(purity_type)
    _tree.set_root(_id3(s, example_weights[t].copy(), None, attributes.copy(), 1))
    return _tree


def _id3(s, _example_weights, parent, _attributes, level):
    """A recursive function that runs the ID3 algorithm. It uses the given purity to split on Attributes."""
    if s[-1].count(s[-1][0]) == len(s[-1]):
        node = Node(s, _example_weights, parent, True)
        node.set_label(s[-1][0])
        return node

    elif len(_attributes) == 0 or level == max_depth:
        node = Node(s, _example_weights, parent, True)
        node.set_label(_find_majority_label(s[-1], _example_weights))
        return node

    else:
        node = Node(s, _example_weights, parent, False)
        _split(node, _get_attributes(_attributes))

        for value in node.attribute:
            arr = _find_s_v(node, node.attribute, value)
            s_v = arr[0]
            _example_weights_v = np.array(arr[1])

            if len(s_v[-1]) == 0:
                label = _find_majority_label(s[-1], _example_weights)
                child = Node({}, np.array([]), node, True)
                child.set_label(label)
                node.add_branch(value, child)

            else:
                a = _attributes.copy()
                a.remove(node.attribute)
                child = _id3(s_v, _example_weights_v, node, a, level + 1)
                node.add_branch(value, child)

        return node


def _find_s_v(node, attribute, value):
    """Finds the subset of examples for a particular attribute value."""
    a_index = attributes.index(attribute)
    indices = [i for i, x in enumerate(node.s[a_index]) if x == value]
    s_v = node.s.copy()

    for i in range(len(node.s)):
        new_feature_list = []

        for index in indices:
            new_feature_list.append(node.s[i][index])
        s_v[i] = new_feature_list

    example_list = []
    for index in indices:
        example_list.append(node.example_weights[index])

    return [s_v, np.array(example_list)]


# Gain
def _split(node, _attributes):
    """Finds the Attribute to split on. Sets the node's attribute."""
    gains = []
    for attribute in _attributes:
        gains.append(_calculate_gain(node, attribute))
    max_index = gains.index(max(gains))
    node.set_attribute(_attributes[max_index])


def _calculate_gain(node, attribute):
    """
    Calculates the gain of the given attribute by using _calculate_purity().
    Gain: Calculates purity of the entire attribute then subtracts the example weight for each attribute value * purity
    of that attribute value.
    """
    gain = 0.0
    gain += _calculate_purity(node.s, node.example_weights)
    for value in attribute:
        arr = _find_s_v(node, attribute, value)
        s_v = arr[0]
        _example_weights_v = np.array(arr[1])

        if len(s_v[-1]) != 0:
            scalar = np.sum(_example_weights_v)
            p = _calculate_purity(s_v, _example_weights_v)
            if p != 0:
                gain -= scalar * p

    return gain


def _calculate_purity(s, _example_weights):
    """Runs the correct purity function based on the input."""
    if   purity_type == "me": return _calculate_majority_error(s, _example_weights)
    elif purity_type == "gi": return _calculate_gini_index(s, _example_weights)
    else: return _calculate_entropy(s, _example_weights)


def _calculate_entropy(s, _example_weights):
    """
    Calculates the entropy by using _find_num_of_s_l() to find the probability of the label.
    Uses probability * log(probability) for every label of the given attribute
    """
    entropy = 0.0
    for label in labels:
        probability_of_label = _find_num_of_s_l(s, label, _example_weights)

        if probability_of_label != 0:
            entropy -= probability_of_label * math.log(probability_of_label, 2)
    return entropy


def _calculate_majority_error(s, _example_weights):
    """
    Calculates the majority error for a given set of examples
    by finding the majority label and calculating the error from that label
    """
    majority_label = _find_majority_label(s[-1], _example_weights)
    me = 1 - _find_num_of_s_l(s, majority_label, _example_weights)
    return me


def _find_majority_label(y, _example_weights):
    """Finds the majority label given a list of label example data"""
    count = [0 for _ in range(len(labels))]
    for i in range(len(y)):
        label = y[i]
        for j in range(len(labels)):
            if label == labels[j]:
                count[j] += _example_weights[i]
                break

    index = count.index(max(count))
    return labels[index]


def _calculate_gini_index(s, _example_weights):
    """
    Calculates the gini index for a given set of examples by subtracting the number of examples for a label
    divided by the total number of examples all squared for every example.
    """
    gi = 1.0
    for label in labels:
        num_of_s_l = _find_num_of_s_l(s, label, _example_weights)
        if num_of_s_l != 0:
            p_l = num_of_s_l
            gi -= p_l**2
    return gi


def _find_num_of_s_l(s, label, _example_weights):
    """Finds the number of examples for a particular label."""
    total = 0.0
    for i in range(len(s[-1])):
        if s[-1][i] == label:
            total += _example_weights[i]
    return total / np.sum(_example_weights)


# Prediction
def calculate_prediction_error_for_tree(s, root):
    """
    Finds the prediction error based on the given data and the given tree (root).
    Calculated by: 1 - correct count / number of examples
    :param s: the entire dataset - [s, y]
    :param root: the root of the learned tree
    :return: percentage of incorrect predictions
    """
    incorrect_count = 0
    for index in range(len(s[-1])):
        example = []
        for l in s:
            example.append(l[index])
        incorrect_count += predict_example(example, root, True)

    return incorrect_count/len(s[-1])


def predict_example(example, node, is_normal):
    """
    A recursive function that predicts the given example.
    :param example: [features] - The feature values for a single example
    :param node: The root node
    :param is_normal: True if running normal ID3 algorithm. False if boosting/bagging
    :return: If is_normal returns 0 if the prediction was correct, 1 otherwise.
            If not is_normal returns the prediction.
    """
    if not node.is_leaf:
        a_index = attributes.index(node.attribute)
        child = node.branches[example[a_index]]
        return predict_example(example, child, is_normal)

    else:
        if is_normal:
            if node.label == example[-1]: return 0
            else: return 1
        else:
            return node.label


# Testing
def _check_tree(node, _attributes=[], branches=[], level=0):
    """A recursive function that walks the tree and prints out the attributes, branches, it took to get to a label."""
    if node.is_leaf:
        _astring = ""
        _bstring = ""
        for _a in _attributes:
            _astring += str(_a) + ", "
        for b in branches:
            _bstring += b + ", "
        print("ATTRIBUTES: ", _astring, "BRANCHES: ", _bstring, "LABEL: ", node.label, "LEVEL: ", level)

    else:
        _attributes.append(node.attribute)
        for branch, child in node.branches.items():
            copy = branches.copy()
            copy.append(branch)
            _check_tree(child, _attributes.copy(), copy, level+1)


# Setup
def setup_data(m=4999, iters=1):
    """
    Sets attributes, labels, example_weights train_data, and test_data based on the data_type
    :param m: number of examples ID3 will run on
    :param iters: number of iterations ID3 will be run
    """
    global attributes, labels, example_weights, train_data, test_data
    if data_type == "car":
        attributes = Import.car_attributes
        labels = Import.car_labels
        m = 1000

    else:
        attributes = Import.bank_attributes
        labels = Import.bank_labels
    example_weights = np.tile(np.repeat(1.0 / m, m), (iters, 1))
    train_data = Import.import_data(data_type, True, True)
    test_data = Import.import_data(data_type, False, True)


def setup_example_ada(T):
    """Sets the global variable up to work with the example data from Import.py"""
    global attributes, labels, example_weights, train_data, test_data
    attributes = Import.example_attributes
    labels = Import.bank_labels
    m = 14
    example_weights = np.tile(np.repeat(1.0/m, m), (T, 1))
    train_data = Import.get_example_data()
    test_data = Import.get_example_data()


if __name__ == '__main__':
    data_type = sys.argv[1]
    purity_type = sys.argv[2]
    if len(sys.argv) > 3:
        max_depth = int(sys.argv[3])

    setup_data()
    _run()

