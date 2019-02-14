import sys
import math

car_attributes = [
    ["vhigh", "high", "med", "low"],
    ["vhigh", "high", "med", "low", "."],
    ["2", "3", "4", "5more"],
    ["2", "4", "more"],
    ["small", "med", "big"],
    ["low", "med", "high"]
]

car_labels = ["unacc", "acc", "good", "vgood"]


def _import_data(path):
    s = []
    with open(path, 'r') as f:
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
    return s


def _get_small_car_example_data():
    return [
        ["low", "med", "high"],
        ["high", "high", "high"],
        ["5more", "5more", "5more"],
        ["4", "4", "4"],
        ["med", "med", "med"],
        ["high", "high", "high"],
        ["vgood", "good", "acc"]
    ]


def _train_data(s, purity, max_depth=-1):
    tree = Tree(purity, max_depth)
    tree.set_root(_id3(s, None, attributes.copy(), purity, 1, max_depth))
    return tree


def _predict(s, root):
    correct_count = 0
    for index in range(len(s[-1])):
        example = []
        i = 0
        for l in s:
            example.append(l[index])
        correct_count += _predict_example(example, root)

    return correct_count/len(s[-1])


def _predict_example(example, node):
    if not node.is_leaf:
        a_index = attributes.index(node.attribute)
        child = node.branches[example[a_index]]
        return _predict_example(example, child)
    else:
        if node.label == example[-1]: return 0
        else: return 1


def _check_tree(node, a, branches, level):
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
            p_l = num_of_s_l / len(s[-1])
            entropy -= p_l * math.log(p_l, 2)
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

    train_data = _import_data("./" + data_type + "/train.csv")
    #train_data = _get_small_car_example_data()
    test_data = _import_data("./" + data_type + "/test.csv")

    attributes = []
    if data_type == "car":
        attributes = car_attributes
        labels = car_labels

    tree = _train_data(train_data, purity=purity_type, max_depth=depth)

    #_check_tree(tree.root, [], [], 1)
    print(_predict(train_data, tree.root))
    print(_predict(test_data, tree.root))

