import statistics
import numpy as np

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
    ["job", "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
        "blue-collar", "self-employed", "retired", "technician", "services"],
    ["marital", "married", "divorced", "single"],
    ["education", "unknown", "secondary", "primary", "tertiary"],
    ["default", "yes", "no"],
    ["numeric", "eunder", "over"],
    ["housing", "yes", "no"],
    ["loan", "yes", "no"],
    ["contact", "unknown", "telephone", "cellular"],
    ["numeric", "eunder", "over"],
    ["month", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    ["numeric", "eunder", "over"],
    ["numeric", "eunder", "over"],
    ["numeric", "eunder", "over"],
    ["numeric", "eunder", "over"],
    ["poutcome", "unknown", "other", "failure", "success"]
]

bank_labels = [-1, 1]

example_attributes = [["s", "o", "r"], ["h", "m", "c"], ["h", "n", "l"], ["s", "w"]]


def import_data(path, train, treat_u_as_value):
    """Imports the data from a csv file into a list of examples."""
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

                if path == "bank" and i == num_columns - 1:
                    if terms[i] == "yes": s[i].append(1)
                    else: s[i].append(-1)

                else: s[i].append(terms[i])
    if path == "bank":
        attributes = bank_attributes
        temp = _change_numeric_attributes_to_binary(s, attributes)
        if not treat_u_as_value:
            s = _change_missing_attributes_to_majority(temp, attributes, train)
    return s


# Numeric Attributes
def _change_numeric_attributes_to_binary(s, attributes):
    """
    Finds all numeric attributes, calculates the median, updates the attributes to contain the median.
    Then updates all examples to contain either "eunder", for equal to or under, or "over" for the numeric attributes.
    """
    for i in range(len(attributes)):
        if attributes[i][0] == "numeric":
            median = _get_median(s[i])
            attributes[i][0] = str(median)
            s[i] = _update_numeric_attributes(s[i], attributes[i])

        elif _is_numeric_attribute(attributes[i]):
            s[i] = _update_numeric_attributes(s[i], attributes[i])
    return s


def _is_numeric_attribute(attribute):
    """Check if a given attribute in the bank list is numeric."""
    try:
        int(attribute[0])
        return True
    except ValueError:
        return False


def _get_median(s_a):
    """Given example values at a numeric attribute, calculates the median of the set."""
    s_a_ints = list(map(int, s_a))  # convert from strings to ints
    median = statistics.median(s_a_ints)
    return median


def _update_numeric_attributes(s_a, attribute):
    for i in range(len(s_a)):
        if int(s_a[i]) > int(attribute[0]): s_a[i] = "over"
        else: s_a[i] = "eunder"
    return s_a


# Missing Attributes
def _change_missing_attributes_to_majority(s, attributes, train):
    majority_attributes = []
    for i in range(len(attributes)):

        if train:
            majority_attributes.append("")
            if "unknown" in attributes[i]:
                majority_attribute = _find_majority_attribute_value(s[i], attributes[i])
                majority_attributes[i] = majority_attribute

                for j in range(len(s[i])):
                    if s[i][j] == "unknown":
                        s[i][j] = majority_attribute

        elif "unknown" in attributes[i]:
            for j in range(len(s[i])):
                if s[i][j] == "unknown":
                    s[i][j] = majority_attributes[i]

    return s


def _find_majority_attribute_value(s_a, attribute):
    """Finds the majority attribute value for the given attribute."""
    count = [0 for _ in range(len(attribute))]

    for value in s_a:
        for i in range(len(attribute)):

            if value == attribute[i] and attribute[i] != "unknown":
                count[i] += 1
                break

    index = count.index(max(count))
    return attribute[index]


def get_small_car_example_data():
    """Gets a list of 3 examples formatted like s."""
    return [
        ["low", "med", "high"],
        ["high", "high", "high"],
        ["5more", "5more", "5more"],
        ["4", "4", "4"],
        ["med", "med", "med"],
        ["high", "high", "high"],
        ["vgood", "good", "acc"]
    ]


def get_small_bank_example_data():
    """Gets a list of 3 examples formatted like s."""
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
    temp = []
    for i in range(len(s[-1])):
        if s[-1][i] == "yes":
            s[-1][i] = 1
        else:
            s[-1][i] = 0
    s = _change_numeric_attributes_to_binary(s, bank_attributes)
    return s


def get_example_data():
    """Gets a list of 4 examples formatted like s."""
    s = [
        ["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"],
        ["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"],
        ["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"],
        ["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"],
        ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"],
    ]
    temp = []
    for i in range(len(s[-1])):
        if s[-1][i] == "yes":
            s[-1][i] = 1
        else:
            s[-1][i] = -1

    return s
