import math
import random
import sys
sys.path.append('../DecisionTree')
import numpy as np
import ID3

T = 1000
m = 4999
trees = []
vote_arr = []
attr_subset_nums = [2, 4, 6]
attr_subset_num = 0


# Adaboost
def _run_ada_boost():
    """Runs T iterations of the Decision Stump ID3 algorithm"""
    global predictions
    test_predictions = np.zeros((T, m))
    y_train = np.array(ID3.train_data[-1])
    y_test = np.array(ID3.test_data[-1])
    train_err_str = ""
    test_err_str = ""

    for t in range(T):
        arr = _train_iter_ada_boost(t, y_train)
        train_err_str += str(_calculate_prediction_error(y_train, arr[0])) + ","

        test_hyp = _test_iter_ada_boost(arr[1], test_predictions)
        test_err_str += str(_calculate_prediction_error(y_test, test_hyp)) + ","

    print("TRAIN: ")
    print(train_err_str)
    print("TEST: ")
    print(test_err_str)

    train_err_str = ""
    test_err_str = ""
    for t in range(T):
        train_err_str += str(ID3.calculate_prediction_error_for_tree(ID3.train_data, trees[t].root)) + ","
        test_err_str += str(ID3.calculate_prediction_error_for_tree(ID3.test_data, trees[t].root)) + ","

    print("DEC STUMP TRAIN: ")
    print(train_err_str)
    print("DEC STUMP TEST: ")
    print(test_err_str)


def _train_iter_ada_boost(t, y):
    """
    Runs one iteration of ada_boost on the train data.
    Trains a decision stump, then calculates the predictions, error and vote, then finds the final hypothesis.
    """
    global vote_arr
    s = ID3.train_data
    tree = ID3.train(s, t)
    trees.append(tree)

    predictions[t] = _calculate_predictions(s, tree.root, predictions[t])
    error = _calculate_error(y, t)

    vote_arr.append(_calculate_vote(error))
    votes = np.array(vote_arr)
    if t != T-1:  _calculate_weights(votes, y, t)

    hyp = _calculate_ada_final_hyp(votes, predictions)
    return [hyp, votes]


def _test_iter_ada_boost(votes, _predictions):
    """Calculates the final hypothesis of the test data when run on the trained trees."""
    s = ID3.test_data
    for t in range(len(trees)):
        _predictions[t] = _calculate_predictions(s, trees[t].root, _predictions[t])
    return _calculate_ada_final_hyp(votes, _predictions)


def _calculate_error(y, t):
    """Calculates the error for predictions[t] with example_weights[t]"""
    return 0.5 - (0.5 * (np.sum(ID3.example_weights[t] * y * predictions[t])))


def _calculate_vote(error):
    """Calculates the vote for the given error"""
    # no base takes natural log
    return 0.5 * math.log((1.0 - error) / error)


def _calculate_weights(votes, y, t):
    """Calculates the weights for the adaboost algorithm"""
    ID3.example_weights[t+1] = ID3.example_weights[t] * np.exp(-votes[t] * y * predictions[t])
    z = np.sum(ID3.example_weights[t+1])
    ID3.example_weights[t+1] /= z


def _calculate_ada_final_hyp(votes, _predictions):
    """Sums up the predictions times the given votes"""
    temp = np.tile(np.zeros(m), (len(votes), 1))
    for index in range(len(votes)):
        temp[index] = np.array(votes[index] * _predictions[index])
    return np.sign(temp.sum(axis=0))


# Bagged Trees
def _run_bagged():
    """Runs the bagged decision tree algorithm for T different samples"""
    global predictions
    predictions = np.zeros(4999)
    test_predictions = np.zeros(4999)
    train_err_str = ""
    test_err_str = ""

    for t in range(T):
        [train_err, predictions] = _run_bagged_iter(t, True, ID3.train_data, predictions)
        train_err_str += str(train_err) + ","

        [test_err, test_predictions] = _run_bagged_iter(t, False, ID3.test_data, test_predictions)
        test_err_str += str(test_err) + ","

    print("TRAIN: ")
    print(train_err_str)
    print("TEST: ")
    print(test_err_str)


def _run_bagged_iter(_t, is_train, data, _predictions):
    """Runs one iteration of the bagged decision tree algorithm"""
    [s, indices] = _draw_sample(data)
    if is_train: trees.append(ID3.train(s, _t, attr_subset_num))
    _predictions = _calculate_bagged_predictions(s, indices, trees[_t].root, _predictions)
    hyp = _calculate_bagged_final_hyp(_predictions)

    return [_calculate_prediction_error(np.array(data[-1], dtype=int), hyp), _predictions]


def _calculate_bagged_final_hyp( _predictions):
    """Updates predictions that are 0 (due to lack of being sampled or even amount of sampling) to -1 or 1"""
    is_even = True
    final_hyp = np.sign(_predictions)
    for i in range(_predictions.size):
        p = _predictions[i]

        if p == 0:
            if is_even: p = 1
            else: p = -1
            is_even = not is_even

        final_hyp[i] = p
    return np.sign(final_hyp)


def _draw_sample(data):
    """Draws m samples uniformly with replacement"""
    s = []
    indices = []
    for i in range(len(data)):
        s.append([])
    for i in range(m):
        n = random.randint(0, len(data[-1]) - 1)
        indices.append(n)
        for j in range(len(data)):
            s[j].append(data[j][n])
    return [s, indices]


# Bias/variance
def _bias_variance_decomp():
    m = 1000
    _trees = []
    for i in range(100):
        trees.append([])

        for t in range(T):
            s = _draw_sample(ID3.train_data)
            trees[i].append(ID3.train(s, t))
            if t == 0: _trees.append(trees[i][t])

    test_predictions = np.zeros((100, m))
    y = np.array(ID3.test_data[-1], dtype=int)
    for i in range(len(_trees)):
        test_predictions[i] = _calculate_predictions(ID3.test_data, _trees[i].root, test_predictions[i])
    # TODO: BIAS
    # bias_terms = 0.0
    # for row in range(len(_trees)):
    # average( h_i(x*)) - f(x*) ) ^2.


# Random Forest
def _run_rand_forest():
    """Runs T iterations of random forest for each attribute subset size in [2, 4, 6]"""
    global attr_subset_num
    for attr_subset_num in attr_subset_nums:
        print("FEATURE_SUBSET_SIZE: ", attr_subset_num, ": _______________________")
        _run_bagged()


# Predictions
def _calculate_predictions(s, root, _predictions):
    """
    Calculates the predictions for the given tree root by using all examples to walk tree
    and using _predict_example()
    """
    p = _predictions.copy()
    for index in range(len(s[-1])):
        example = []
        for l in s:
            example.append(l[index])
        p[index] = ID3.predict_example(example, root, False)
    return p


def _calculate_bagged_predictions(s, indices, root, _predictions):
    """Uses the initial indexes of the sample data to calculate the overall prediction of all examples in s"""
    round_predictions = np.zeros(len(indices))
    round_predictions = _calculate_predictions(s, root, round_predictions)
    for i in range(len(indices)):
        _predictions[indices[i]] = predictions[indices[i]] + round_predictions[i]
    return _predictions


def _calculate_prediction_error(y, _predictions):
    """Calculates the percentage of incorrect predictions"""
    count = 0
    for i in range(len(y)):
        if y[i] != _predictions[i]: count += 1
    return count / len(y)


if __name__ == '__main__':
    ID3.data_type = "bank"
    alg_type = sys.argv[1]

    if alg_type == "ada":
        ID3.max_depth = 2
        predictions = np.zeros((T, m))
    else:
        predictions = np.zeros(4999)
        m = 2500

    ID3.setup_data(m, T)

    if alg_type == "ada":
        print("ADA BOOST")
        _run_ada_boost()

    elif alg_type == "bag":
        print("BAGGED DECISION TREES")
        _run_bagged()

    elif alg_type == "forest":
        print("RANDOM FOREST")
        _run_rand_forest()
