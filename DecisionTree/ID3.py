import sys
import math
import Import
import numpy as np

# Variables
alg_type = "normal"
data_type = "bank"
purity_type = "ig"
max_depth = -1
m = 5000
T = 100
example_weights = np.tile(np.repeat(1.0/m, m-1), (T+1, 1))
predictions = np.empty((T+1, m-1))


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


# Train
def _run_normal():
    """Runs the normal ID3 algorithm"""
    # example_data = Import.get_small_bank_example_data()
    tree = _train_data(train_data, 0)
    #_check_tree(tree.root)
    print("TRAIN: ", _calculate_error(train_data, tree.root))
    #print("TEST: ", _calculate_error(test_data, tree.root))


def _run_ada_boost():
    """Runs 1000 iterations of the Decision Stump ID3 algorithm"""
    # TODO: 1st round is good. Run through gain on second and rest of ada
    global example_weights, predictions
    votes = np.empty(T+1)
    y = np.array(train_data[-1])
    z = np.zeros(T+1)

    for t in range(0, T):
        tree = _train_data(train_data, t)
        _calculate_ada_predictions(train_data, t, tree.root)
        error = 0.5 - (0.5 * (np.sum(example_weights[t] * y * predictions[t])))

        vote = _calculate_vote(error)
        votes[t] = vote

        example_weights[t+1] = example_weights[t] * np.exp(-vote * y * predictions[t])
        z[t] = np.sum(example_weights[t+1])
        example_weights[t+1] /= z[t]

    tree = _train_data(train_data, T)
    _calculate_ada_predictions(train_data, T, tree.root)
    error = 0.5 - (0.5 * (np.sum(example_weights[T] * y * predictions[T])))

    vote = _calculate_vote(error)
    votes[t] = vote

    temp = np.tile(np.empty(m-1), (T+1, 1))
    for index in range(len(votes)):
        temp[index] = np.array(votes[index] * predictions[index])
    final_hyp = np.sign(temp.sum(axis=0))
    print(final_hyp)


def _calculate_vote(error):
    # no base takes natural log
    return 0.5 * math.log((1.0 - error) / error)


def _train_data(s, t):
    """Trains a decision tree with the given data, the ID3 algorithm, and the type of purity function given."""
    _tree = Tree(purity_type)
    _tree.set_root(_id3(s, example_weights[t].copy(), None, attributes.copy(), 1))
    return _tree


# ID3
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
        _split(node, _attributes)

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
    for i in range(len(_attributes)):
        gains.append(_calculate_gain(node, _attributes[i]))
    max_index = gains.index(max(gains))

    node.set_attribute(_attributes[max_index])


def _calculate_gain(node, attribute):
    """ """
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
    """ """
    entropy = 0.0
    for label in labels:
        num_of_s_l = _find_num_of_s_l(s, label, _example_weights)
        if num_of_s_l != 0:
            probability_of_label = num_of_s_l
            entropy -= probability_of_label * math.log(probability_of_label, 2)
    return entropy


def _calculate_majority_error(s, _example_weights):
    """
    Calculates the majority error for a given set of examples
    by finding the majority label and calculating the error from that label
    """
    majority_label = _find_majority_label(s[-1])
    me = 1 - _find_num_of_s_l(s, majority_label, _example_weights)
    return me


def _find_majority_label(s_labels, _example_weights):
    """Finds the majority label given a list of label example data"""
    count = [0 for _ in range(len(labels))]
    for i in range(len(s_labels)):
        label = s_labels[i]
        for j in range(len(labels)):
            if label == labels[j]:
                count[j] += _example_weights[i]
                break

    index = count.index(max(count))
    return labels[index]


def _calculate_gini_index(s, _example_weights):
    """Calculates the gini index for a given set of examples.
     by subtracting the number of examples for a label divided by the total number of examples all squared for every example.
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
            total += 1
    return total / _example_weights.size


# Prediction
def _calculate_error(s, root):
    """
    Finds the prediction error based on the given data and the given tree (root).
    Calculated by: 1 - correct count / number of examples
    """
    correct_count = 0
    for index in range(len(s[-1])):
        example = []
        for l in s:
            example.append(l[index])
        correct_count += _predict_example(example, root, False)

    return correct_count/len(s[-1])


def _calculate_ada_predictions(s, t, root):
    """ """
    error = 0
    global predictions
    for index in range(len(s[-1])):
        example = []
        for l in s:
            example.append(l[index])
        prediction = _predict_example(example, root, True)

        predictions[t, index] = prediction


def _predict_example(example, node, is_ada):
    """A recursive function that predicts the given example. Then returns whether the prediction was correct or not."""
    if not node.is_leaf:
        a_index = attributes.index(node.attribute)
        child = node.branches[example[a_index]]
        return _predict_example(example, child, is_ada)
    else:
        if not is_ada:
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
        # print(node.branches.items())
        for branch, child in node.branches.items():
            copy = branches.copy()
            copy.append(branch)
            _check_tree(child, _attributes.copy(), copy, level+1)


# Setup
def _setup():
    """Sets the global variables based on the sys arguments."""
    global max_depth, alg_type
    if sys.argv[1] == "ada":
        alg_type = "ada"
        max_depth = 2

    # Normal Decision Tree
    else:
        global data_type, purity_type
        data_type = sys.argv[1]
        purity_type = sys.argv[2]
        if len(sys.argv) > 3:
            max_depth = int(sys.argv[3])
    _set_attributes()


def _set_attributes():
    """Sets the attributes and labels based on the data_type"""
    global attributes, labels, m, example_weights
    if data_type == "car":
        attributes = Import.car_attributes
        labels = Import.car_labels
        m = 1000
        example_weights = np.tile(np.repeat(1.0 / m, m-1), (1000, 1))

    else:
        attributes = Import.bank_attributes
        labels = Import.bank_labels


if __name__ == '__main__':
    _setup()
    # attributes = Import.example_attributes
    # labels = Import.bank_labels
    # m = 14
    # example_weights = np.tile(np.repeat(1.0 / m, m), (T+1, 1))
    # predictions = np.empty((T+1, m))
    # train_data = Import.get_example_data()
    train_data = Import.import_data(data_type, True, True)
    test_data = Import.import_data(data_type, False, True)
    if alg_type == "ada":
        _run_ada_boost()
    else:
        _run_normal()

