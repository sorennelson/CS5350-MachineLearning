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

majority_attributes = []


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
        attributes = bank_attributes
        temp = _change_numeric_attributes_to_binary(s, attributes)
        s = _change_missing_attributes_to_majority(temp, attributes, train)
    return s


# Numeric Attributes
def _change_numeric_attributes_to_binary(s, attributes):
    """
    Finds all numeric attributes, calculates the median, updates the attributes to contain the median.
    Then updates all examples to contain either "eunder", for equal to or under, or "over" for the numeric attributes.

    :param s: the entire dataset
    :param attributes: all attributes
    :return: the updated dataset
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
    """
    Given example values at a numeric attribute, calculates the median of the set.

    :param s_a: example values at a numeric attibute
    :return: the median
    """
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
    """
    Finds the majority attribute value for the given attribute.

    :param s_a: The data for an attribute
    :param attribute: The attribute to find the majority of
    :return: The majority attribute value
    """
    count = [0 for _ in range(len(attribute))]

    for value in s_a:
        for i in range(len(attribute)):

            if value == attribute[i] and attribute[i] != "unknown":
                count[i] += 1
                break

    index = count.index(max(count))
    return attribute[index]


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