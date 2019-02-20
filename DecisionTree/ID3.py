import sys
import math
import Import

# Variables
alg_type = "normal"
data_type = "bank"
purity_type = "ig"
max_depth = -1
attributes = []
example_weights = []


# Structures
class Tree:
    def __init__(self, purity="ig", max_depth=0):
        self.purity = purity
        self.max_depth = max_depth
        self.root = None

    def set_root(self, node):
        self.root = node


class Node:
    def __init__(self, s, parent, is_leaf):
        self.s = s
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
    tree = _train_data(train_data)
    print("TRAIN: ", _calculate_error(train_data, tree.root))
    print("TEST: ", _calculate_error(test_data, tree.root))


def _run_ada_boost():
    """Runs 1000 iterations of the Decision Stump ID3 algorithm"""
    trees = []
    votes = []

    for t in range(1, 1001):
        trees.append(_train_data(train_data))
        error = _calculate_ada_error(train_data, trees[-1].root)

        # no base takes natural log
        vote = 0.5 * math.log((1-error)/error)
        z = 0
        for index in range(len(example_weights)):
            global example_weights
            # example_weights[index] =




def _calculate_weights():
    """ """


def _train_data(s):
    """Trains a decision tree with the given data, the ID3 algorithm, and the type of purity function given."""
    _tree = Tree(purity_type, max_depth)
    _tree.set_root(_id3(s, None, attributes.copy(), 1))
    return _tree


# ID3
def _id3(s, parent, _attributes, level):
    """A recursive function that runs the ID3 algorithm. It uses the given purity to split on Attributes."""
    if s[-1].count(s[-1][0]) == len(s[-1]):
        node = Node(s, parent, True)
        node.set_label(s[-1][0])
        return node

    elif len(_attributes) == 0 or level == max_depth:
        node = Node(s, parent, True)
        node.set_label(_find_majority_label(s[-1]))
        return node

    else:
        node = Node(s, parent, False)
        _split(node, _attributes)

        for value in node.attribute:
            s_v = _find_s_v(node, node.attribute, value)

            if len(s_v[-1]) == 0:
                label = _find_majority_label(s[-1])
                child = Node({}, node, True)
                child.set_label(label)
                node.add_branch(value, child)

            else:
                a = _attributes.copy()
                a.remove(node.attribute)
                child = _id3(s_v, node, a, level + 1, max_depth)
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

    return s_v


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
    gain += _calculate_purity(node.s)
    for value in attribute:
        s_v = _find_s_v(node, attribute, value)

        if len(s_v[-1]) != 0:
            scalar = len(s_v[-1]) / len(node.s[-1])
            p = _calculate_purity(s_v)

            if p != 0:
                gain -= scalar * p

    return gain


def _calculate_purity(s):
    """Runs the correct purity function based on the input."""
    if   purity_type == "me": return _calculate_majority_error(s)
    elif purity_type == "gi": return _calculate_gini_index(s)
    else: return _calculate_entropy(s)


def _calculate_entropy(s):
    """ """
    entropy = 0.0
    for label in labels:
        num_of_s_l = _find_num_of_s_l(s, label)
        if num_of_s_l != 0:
            probability_of_label = num_of_s_l / len(s[-1])
            entropy -= probability_of_label * math.log(probability_of_label, 2)
    return entropy


def _calculate_majority_error(s):
    """
    Calculates the majority error for a given set of examples
    by finding the majority label and calculating the error from that label
    """
    majority_label = _find_majority_label(s[-1])
    me = 1 - _find_num_of_s_l(s, majority_label) / len(s[-1])
    return me


def _find_majority_label(s_labels):
    """Finds the majority label given a list of label example data"""
    count = [0 for _ in range(len(labels))]
    for label in s_labels:
        for i in range(len(labels)):
            if label == labels[i]:
                count[i] += 1
                break

    index = count.index(max(count))
    return labels[index]


def _calculate_gini_index(s):
    """Calculates the gini index for a given set of examples.
     by subtracting the number of examples for a label divided by the total number of examples all squared for every example.
     """
    gi = 1.0
    for label in labels:
        num_of_s_l = _find_num_of_s_l(s, label)
        if num_of_s_l != 0:
            p_l = num_of_s_l / len(s[-1])
            gi -= p_l**2
    return gi


def _find_num_of_s_l(s, label):
    """Finds the number of examples for a particular label."""
    return len([i for i, x in enumerate(s[-1]) if x == label])


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
        correct_count += _predict_example(example, root)

    return correct_count/len(s[-1])


def _calculate_ada_error(s, root):
    """ """
    error = 0
    for index in range(len(s[-1])):
        example = []
        for l in s:
            example.append(l[index])
        if _predict_example(example, root) == 0:
            error += example_weights

    return error


def _predict_example(example, node):
    """A recursive function that predicts the given example. Then returns whether the prediction was correct or not."""
    if not node.is_leaf:
        a_index = attributes.index(node.attribute)
        child = node.branches[example[a_index]]
        return _predict_example(example, child)
    else:
        if node.label == example[-1]: return 0
        else: return 1


# Testing
def _check_tree(node, _attributes=[], branches=[], level=0):
    """A recursive function that walks the tree and prints out the attributes, branches, it took to get to a label."""
    if node.is_leaf:
        astring = ""
        bstring = ""
        for i in a:
            astring += str(i) + ", "
        for b in branches:
            bstring += b + ", "
        print("ATTRIBUTES: ", astring, "BRANCHES: ", bstring, "LABEL: ", node.label, "LEVEL: ", level)

    else:
        _attributes.append(node.attribute)
        for branch, child in node.branches.items():
            copy = branches.copy()
            copy.append(branch)
            _check_tree(child, _attributes.copy(), copy, level+1)


# Setup
def _setup():
    """Sets the global variables based on the sys arguments"""
    global max_depth, alg_type, example_weights
    if sys.argv[1] == "ada":
        alg_type = "ada"
        max_depth = 2
        example_weights = Import.get_initial_example_weights(5000)

    # Normal Decision Tree
    else:
        global data_type, purity_type
        data_type = sys.argv[1]
        purity_type = sys.argv[2]
        if len(sys.argv) > 3:
            max_depth = int(sys.argv[3])
        else:
            max_depth = -1
    _set_attributes()


def _set_attributes():
    """Sets the attributes and labels based on the data_type"""
    global attributes, labels
    if data_type == "car":
        attributes = Import.car_attributes
        labels = Import.car_labels

    else:
        attributes = Import.bank_attributes
        labels = Import.bank_labels


if __name__ == '__main__':
    _setup()
    train_data = Import.import_data(data_type, True, True)
    test_data = Import.import_data(data_type, False, True)
    if alg_type == "ada":
        _run_ada_boost()
    else:
        _run_normal()

