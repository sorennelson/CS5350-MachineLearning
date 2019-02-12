from enum import Enum
import csv
import sys
import math


def _import_data(data):
    s = {
        "buying": [],
        "maintenance": [],
        "doors": [],
        "persons": [],
        "lug_boot": [],
        "safety": [],
        "label": []
    }
    path = ""
    if data == "train":
        path = './car/train.csv'
    else:
        path = './car/test.csv'

    with open(path, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            s["buying"].append(terms[0])
            s["maintenance"].append(terms[1])
            s["doors"].append(terms[2])
            s["persons"].append(terms[3])
            s["lug_boot"].append(terms[4])
            s["safety"].append(terms[5])
            s["label"].append(terms[6])
    return s


def _train_data(s, purity='ig', max_depth=0):
    tree = Tree(purity, max_depth)
    attributes = [Attribute.buying, Attribute.maintenance, Attribute.doors,
                  Attribute.persons, Attribute.lug_boot, Attribute.safety]
    tree.set_root(_id3(s, None, attributes, purity, 1, max_depth))
    print("DONE")
    return tree


def _predict_train(s, root):
    prediction = 0
    for index in range(len(s["label"])):
        example = {
            "buying": s["buying"][index],
            "maintenance": s["maintenance"][index],
            "doors": s["doors"][index],
            "persons": s["persons"][index],
            "lug_boot": s["lug_boot"][index],
            "safety": s["safety"][index],
            "label": s["label"][index]
        }
        prediction += _predict_example(example, root)

    return prediction/len(s["label"])


def _predict_test():
    return


def _predict_example(example, node):
    if not node.is_leaf:
        child = node.branches[example[node.attribute.name]]
        return _predict_example(example, child)
    else:
        if node.label == example["label"]: return 1
        else: return 0


def _check_tree(node, attributes, branches, level):
    if node.is_leaf:
        astring = ""
        bstring = ""
        for a in attributes:
            astring += a + ", "
        for b in branches:
            bstring += b + ", "
        print("ATTRIBUTES: ", astring, "BRANCHES: ", bstring, "LABEL: ", node.label, "LEVEL: ", level)

    else:
        attributes.append(node.attribute.name)
        for branch, child in node.branches.items():
            copy = branches.copy()
            copy.append(branch)
            _check_tree(child, attributes.copy(), copy, level+1)


def _id3(s, parent, attributes, purity, level, max_depth):
    if s["label"].count(s["label"][0]) == len(s):
        node = Node(s, parent, True)
        node.set_label(s["label"][0])
        return node

    elif len(attributes) == 0 or level == max_depth:
        node = Node(s, parent, True)
        node.set_label(_find_majority_label(s["label"]))
        return node

    else:
        node = Node(s, parent, False)
        _split(purity, node, attributes)

        for value in node.attribute.value:
            s_v = _find_s_v(s, node.attribute, value)

            if len(s_v["label"]) == 0:
                label = _find_majority_label(s["label"])
                child = Node({}, node, True)
                child.set_label(label)
                node.add_branch(value, child)

            else:
                copy = attributes.copy()
                copy.remove(node.attribute)
                child = _id3(s_v, node, copy, purity, level + 1, max_depth)
                node.add_branch(value, child)

        return node


def _find_majority_label(all_labels):
    labels = ["unacc", "acc", "good", "vgood"]
    count = [0, 0, 0, 0]
    for label in all_labels:
        if   label == labels[0]: count[0] += 1
        elif label == labels[1]: count[1] += 1
        elif label == labels[2]: count[2] += 1
        elif label == labels[3]: count[3] += 1
    index = count.index(max(count))
    return labels[index]


def _find_s_v(s, attribute, value):
    indices = [i for i, x in enumerate(s[attribute.name]) if x == value]
    s_v = s.copy()

    for key, feature_list in s.items():
        new_feature_list = []

        for index in indices:
            new_feature_list.append(feature_list[index])
        s_v[key] = new_feature_list

    return s_v


def _find_num_of_s_l(s, label):
    return len([i for i, x in enumerate(s["label"]) if x == label])


def _split(purity, node, attributes):
    gains = []
    for i in range(len(attributes)):
        gains.append(_calculate_gain(node.s, attributes[i], purity))
    index = gains.index(max(gains))
    node.set_attribute(attributes[index])


def _calculate_gain(s, attribute, purity):
    gain = 0.0
    gain += _calculate_purity(s, purity)
    for value in attribute.value:
        s_v = _find_s_v(s, attribute, value)
        if len(s_v["label"]) != 0:
            scalar = len(s_v["label"]) / len(s["label"])
            purity = _calculate_purity(s_v, purity)
            if purity != 0:
                gain -= scalar * purity
    return gain


def _calculate_purity(s, purity):
    if   purity == "me": return _calculate_majority_error(s)
    elif purity == "gi": return _calculate_gini_index(s)
    else: return _calculate_entropy(s)


def _calculate_entropy(s):
    labels = ["unacc", "acc", "good", "vgood"]
    entropy = 0.0
    for label in labels:
        num_of_s_l = _find_num_of_s_l(s, label)
        if num_of_s_l == 0:
            return 0.0

        p_l = num_of_s_l / len(s["label"])
        entropy -= p_l * math.log(p_l, 2)
    return entropy


def _calculate_majority_error(s):
    majority_label = _find_majority_label(s["label"])
    me = 1 - _find_num_of_s_l(s, majority_label) / len(s["label"])
    return me


def _calculate_gini_index(s):
    labels = ["unacc", "acc", "good", "vgood"]
    gi = 1.0
    for label in labels:
        num_of_s_l = _find_num_of_s_l(s, label)
        if num_of_s_l != 0:
            p_l = num_of_s_l / len(s["label"])
            gi -= p_l**2
    return gi


class Attribute(Enum):
    default = []
    buying = ["vhigh", "high", "med", "low"]
    maintenance = ["vhigh", "high", "med", "low", "."]
    doors = ["2", "3", "4", "5more"]
    persons = ["2", "4", "more"]
    lug_boot = ["small", "med", "big"]
    safety = ["low", "med", "high"]


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
        self.attribute = Attribute.default
        self.label = None

    def set_attribute(self, attribute):
        self.attribute = attribute

    def set_label(self, label):
        self.label = label

    def add_branch(self, value, node):
        self.branches[value] = node


if __name__ == '__main__':
    train_data = _import_data("train")
    tree = None

    if len(sys.argv) == 2:
        try:
            depth = int(sys.argv[1])
            tree = _train_data(train_data, max_depth=depth)
        except ValueError:
            purity_type = sys.argv[1]
            tree = _train_data(train_data, purity=purity_type)

    elif (len(sys.argv)) == 3:
        try:
            depth = int(sys.argv[1])
            purity_type = sys.argv[2]
            tree = _train_data(train_data, max_depth=depth, purity=purity_type)
        except ValueError:
            purity_type = sys.argv[1]
            depth = int(sys.argv[2])
            tree = _train_data(train_data, max_depth=depth, purity=purity_type)

    else:
        tree = _train_data(train_data)
        
    print(_predict_train(train_data, tree.root))
    # example1 = {
    #     "buying": "low",
    #     "maintenance": "high",
    #     "doors": "5more",
    #     "persons": "4",
    #     "lug_boot": "med",
    #     "safety": "high",
    #     "label": "vgood"
    # }
    # print(_predict_example(example1, tree.root))
