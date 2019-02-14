import sys
import math
import statistics

car_attributes = [
    ["vhigh", "high", "med", "low"],
    ["vhigh", "high", "med", "low", "."],
    ["2", "3", "4", "5more"],
    ["2", "4", "more"],
    ["small", "med", "big"],
    ["low", "med", "high"]
]

car_labels = ["unacc", "acc", "good", "vgood"]

bank_attributes = [
    ["numeric", "eunder", "over"],
    ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
        "blue-collar", "self-employed", "retired", "technician", "services"],
    ["married", "divorced", "single"],
    ["unknown", "secondary", "primary", "tertiary"],
    ["yes", "no"],
    ["numeric", "eunder", "over"],
    ["yes", "no"],
    ["yes", "no"],
    ["unknown", "telephone", "cellular"],
    ["numeric", "eunder", "over"],
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    ["numeric", "eunder", "over"],
    ["numeric", "eunder", "over"],
    ["numeric", "eunder", "over"],
    ["numeric", "eunder", "over"],
    ["unknown", "other", "failure", "success"]
]

bank_labels = ["yes", "no"]


def _import_data(path, train):
    """
    Imports the data from a csv file into a list of examples.

    :param path: the data type to import. Either car or bank.
    :return: the data as a list of lists that contain all the example values for an attribute or label (at s[-1]).
    """
    s = []
    fp = "./" + path
    if train:
        fp = path + "/train.csv"
    else:
        fp = path + "/test.csv"
    with open(fp, 'r') as f:
        num_columns = 0
        for line in f:
            terms = line.strip().split(',')
            num_columns = len(terms)
            break

        s = [[] for _ in range(num_columns)]

        for line in f:
            terms = line.strip().split(',')
            for i in range(num_columns):
                s[i].append(terms[i])
    if path == "bank":
        s = _change_numeric_attributes_to_binary(s)
    return s


def _change_numeric_attributes_to_binary(s):
    for i in range(len(attributes)):

        if attributes[i][0] == "numeric":
            median = _get_median(s[i])
            attributes[i][0] = str(median)
            s[i] = _update_numeric_attributes(s[i], attributes[i])

        elif _is_numeric_attribute(attributes[i]):
            s[i] = _update_numeric_attributes(s[i], attributes[i])
    return s


def _is_numeric_attribute(attribute):
    """
    Check if a given attribute in the bank list is numeric
    :param attribute:
    :return: Boolean
    """
    try:
        int(attribute[0])
        return True
    except ValueError:
        return False



def _get_median(s_a):
    s_a_ints = list(map(int, s_a))  # convert from strings to ints
    median = statistics.median(s_a_ints)
    return median


def _update_numeric_attributes(s_a, attribute):
    for i in range(len(s_a)):
        if int(s_a[i]) < int(attribute[0]): s_a[i] = "over"
        else: s_a[i] = "eunder"
    return s_a


def _get_small_car_example_data():
    """
    Gets a list of 3 examples formatted like s.

    :return: the examples
    """
    return [
        ["low", "med", "high"],
        ["high", "high", "high"],
        ["5more", "5more", "5more"],
        ["4", "4", "4"],
        ["med", "med", "med"],
        ["high", "high", "high"],
        ["vgood", "good", "acc"]
    ]


def _get_small_bank_example_data():
    s = [
        ["48","48","53"],
        ["services","blue-collar","technician"],
        ["married", "married", "married"],
        ["secondary", "secondary", "secondary"],
        ["no", "no", "no"],
        ["0", "0", "0"],
        ["yes", "yes", "yes"],
        ["no", "no", "no"],
        ["unknown", "unknown", "unknown"],
        ["5", "5", "5"],
        ["may", "may", "may"],
        ["114", "114", "114"],
        ["2", "2", "2"],
        ["-1", "-1", "-1"],
        ["0", "0", "0"],
        ["unknown", "unknown", "unknown"],
        ["no", "no", "yes"],
    ]
    s = _change_numeric_attributes_to_binary(s)
    return s


def _train_data(s, purity, max_depth=-1):
    """
    Trains a decision tree with the given data, the ID3 algorithm, and the type of purity function given.

    :param s: The imported data.
    :param purity: The type of purity function to use.
                    "ig" for information gain. "me" for Majority Error. "gi" for Gini Index.
    :param max_depth: The max depth of the tree. -1 sets no bounds.
    :return: The tree created.by the ID3 algorithm.
    """
    tree = Tree(purity, max_depth)
    tree.set_root(_id3(s, None, attributes.copy(), purity, 1, max_depth))
    return tree


def _predict(s, root):
    """
    Finds the prediction error based on the given data and the given tree (root). Calculated by: 1 - correct count / number of examples

    :param s: The imported data.
    :param root: The root of the learned tree.
    :return: the prediction error.
    """
    correct_count = 0
    for index in range(len(s[-1])):
        example = []
        for l in s:
            example.append(l[index])
        correct_count += _predict_example(example, root)

    return correct_count/len(s[-1])


def _predict_example(example, node):
    """
    A recursive function that predicts the given example. Then returns whether it was correct or not

    :param example: A single example from the data.
    :param node: The current node to either split on or predict with.
    :return: 0 - correct. 1 - incorrect
    """
    if not node.is_leaf:
        # print(node.attribute)
        a_index = attributes.index(node.attribute)
        # print(a_index)
        # print(example[a_index])
        child = node.branches[example[a_index]]
        return _predict_example(example, child)
    else:
        if node.label == example[-1]: return 0
        else: return 1


def _check_tree(node, a=[], branches=[], level=1):
    """
    A recursive function that walks the tree and prints out the attributes, branches, it took to get to a label.

    :param node: pass in the root node of the trained tree
    """
    if node.is_leaf:
        astring = ""
        bstring = ""
        for i in a:
            astring += str(i) + ", "
        for b in branches:
            bstring += b + ", "
        print("ATTRIBUTES: ", astring, "BRANCHES: ", bstring, "LABEL: ", node.label, "LEVEL: ", level)

    else:
        a.append(node.attribute)
        for branch, child in node.branches.items():
            copy = branches.copy()
            copy.append(branch)
            _check_tree(child, a.copy(), copy, level+1)


def _id3(s, parent, a, purity, level, max_depth):
    if s[-1].count(s[-1][0]) == len(s[-1]):
        node = Node(s, parent, True)
        node.set_label(s[-1][0])
        return node

    elif len(a) == 0 or level == max_depth:
        node = Node(s, parent, True)
        node.set_label(_find_majority_label(s[-1]))
        return node

    else:
        node = Node(s, parent, False)
        _split(purity, node, a)

        for value in node.attribute:
            s_v = _find_s_v(node, node.attribute, value)

            if len(s_v[-1]) == 0:
                label = _find_majority_label(s[-1])
                child = Node({}, node, True)
                child.set_label(label)
                node.add_branch(value, child)

            else:
                copy = a.copy()
                copy.remove(node.attribute)
                child = _id3(s_v, node, copy, purity, level + 1, max_depth)
                node.add_branch(value, child)

        return node


def _find_majority_label(s_labels):
    count = [0 for _ in range(len(labels))]
    for label in s_labels:
        for i in range(len(labels)):
            if label == labels[i]:
                count[i] += 1
                break

    index = count.index(max(count))
    return labels[index]


def _find_s_v(node, attribute, value):
    a_index = attributes.index(attribute)
    indices = [i for i, x in enumerate(node.s[a_index]) if x == value]
    s_v = node.s.copy()

    for i in range(len(node.s)):
        new_feature_list = []

        for index in indices:
            new_feature_list.append(node.s[i][index])
        s_v[i] = new_feature_list

    return s_v


def _find_num_of_s_l(s, label):
    return len([i for i, x in enumerate(s[-1]) if x == label])


def _split(purity, node, a):
    gains = []
    for i in range(len(a)):
        gains.append(_calculate_gain(node, a[i], purity))
    max_index = gains.index(max(gains))

    node.set_attribute(a[max_index])


def _calculate_gain(node, attribute, purity):
    gain = 0.0
    gain += _calculate_purity(node.s, purity)
    #print(attribute)
    for value in attribute:
        s_v = _find_s_v(node, attribute, value)

        if len(s_v[-1]) != 0:
            scalar = len(s_v[-1]) / len(node.s[-1])
            p = _calculate_purity(s_v, purity)

            if p != 0:
                gain -= scalar * p

    return gain


def _calculate_purity(s, purity):
    if   purity == "me": return _calculate_majority_error(s)
    elif purity == "gi": return _calculate_gini_index(s)
    else: return _calculate_entropy(s)


def _calculate_entropy(s):
    entropy = 0.0
    for label in labels:
        num_of_s_l = _find_num_of_s_l(s, label)
        if num_of_s_l != 0:
            probability_of_label = num_of_s_l / len(s[-1])
            entropy -= probability_of_label * math.log(probability_of_label, 2)
    return entropy


def _calculate_majority_error(s):
    majority_label = _find_majority_label(s[-1])
    me = 1 - _find_num_of_s_l(s, majority_label) / len(s[-1])
    return me


def _calculate_gini_index(s):
    gi = 1.0
    for label in labels:
        num_of_s_l = _find_num_of_s_l(s, label)
        if num_of_s_l != 0:
            p_l = num_of_s_l / len(s[-1])
            gi -= p_l**2
    return gi


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


if __name__ == '__main__':
    data_type = sys.argv[1]
    purity_type = sys.argv[2]

    if len(sys.argv) > 3:
        depth = int(sys.argv[3])
    else:
        depth = -1

    attributes = []
    if data_type == "car":
        attributes = car_attributes
        labels = car_labels
    else:
        attributes = bank_attributes
        labels = bank_labels

    train_data = _import_data(data_type, True)
    #train_data = _get_small_car_example_data()
    #train_data = _get_small_bank_example_data()
    test_data = _import_data(data_type, False)

    tree = _train_data(train_data, purity=purity_type, max_depth=depth)

    #print(train_data)

    #_check_tree(tree.root, [], [], 1)
    print(_predict(train_data, tree.root))
    print(_predict(test_data, tree.root))

