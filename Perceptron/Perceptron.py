import sys
import numpy as np


# Standard
def _run_standard_algorithm(learning_rate=1.0):
    """
    Runs 10 epochs of the Standard Perceptron Algorithm.
    Calculates and prints the trian error and the test error at each epoch.
    """
    weights = np.zeros((1, 4), dtype="float")
    weights = np.asmatrix(weights)

    for epoch in range(10):
        weights = _run_standard_epoch(weights, learning_rate)

        train_predictions = np.sign(weights * np.transpose(train_data))
        train_error = _calculate_error(train_predictions, train_y)

        test_predictions = np.sign(weights * np.transpose(test_data))
        test_error = _calculate_error(test_predictions, test_y)

        print(epoch, "- Train Error:", train_error, " Test Error:", test_error)

    print("Final Weights: ", weights)


def _run_standard_epoch(weights, learning_rate):
    """Runs a single epoch of the Standard Perceptron Algorithm."""
    [y, data] = _shuffle_data(train_y, train_data)

    for i in range(y.size):
        [y_example, x_example] = _get_example(i, y, data)

        if _is_error(weights, y_example, x_example):
            weights = _update_weights(weights, y_example, x_example, learning_rate)

    return weights


def _shuffle_data(y, data):
    """Shuffles the given data by appending y to the data, then shuffling, then returns the separated data and y."""
    combined = np.c_[data.reshape(len(data), -1), y.reshape(len(y), -1)]
    np.random.shuffle(combined)
    shuffled_data = combined[:, :data.size // len(data)].reshape(data.shape)
    shuffled_y = combined[:, data.size // len(data):].reshape(y.shape)
    return [shuffled_y, shuffled_data]


def _run_voted_algorithm(learning_rate=1.0):
    """
    Runs 10 epochs of the Voted Perceptron Algorithm.
    Calculates and prints the trian error and the test error at each epoch.
    """
    weights = np.zeros((1, 4), dtype="float")
    weights = np.asmatrix(weights)
    m = 0
    votes = [0.0]

    for epoch in range(10):
        [weights, votes, m] = _run_voted_epoch(weights, votes, learning_rate, m)

        train_predictions = _calculate_voted_predictions(weights, votes, train_data, len(train_y))
        train_error = _calculate_error(train_predictions, train_y)

        test_predictions = _calculate_voted_predictions(weights, votes, test_data, len(test_y))
        test_error = _calculate_error(test_predictions, test_y)
        print(epoch, "- Train Error:", train_error, " Test Error:", test_error)

    print("Votes: ", votes)
    print("Final Weights: ", weights)


def _run_voted_epoch(weights, votes, learning_rate, m):
    """
    Runs 10 epochs of the Voted Perceptron Algorithm.
    Calculates and prints the trian error and the test error at each epoch.
    """
    for i in range(train_y.size):
        [y_example, x_example] = _get_example(i, train_y, train_data)

        if _is_error(weights[m], y_example, x_example):
            new_row = _update_weights(weights[m], y_example, x_example, learning_rate)
            weights = np.r_[weights, new_row]
            votes.append(1)
            m += 1
        else:
            votes[m] += 1

    return [weights, votes, m]


def _calculate_voted_predictions(weights, votes, data, y_len):
    """Runs a single epoch of the Average Perceptron Algorithm"""
    voted_predictions = np.zeros((1, y_len), dtype=float)
    for i in range(len(votes)):
        prediction = np.sign(weights[i] * np.transpose(data))
        voted_predictions += votes[i] * prediction
    return np.sign(voted_predictions)


def _run_average_algorithm(learning_rate=1.0):
    """
    Runs 10 epochs of the Average Perceptron Algorithm.
    Calculates and prints the train error and the test error at each epoch.
    """
    weights = np.zeros((1, 4), dtype="float")
    weights = np.asmatrix(weights)
    average = weights.copy()

    for epoch in range(10):
        [average, weights] = _run_average_epoch(weights, average, learning_rate)

        train_predictions = np.sign(average * np.transpose(train_data))
        train_error = _calculate_error(train_predictions, train_y)

        test_predictions = np.sign(weights * np.transpose(test_data))
        test_error = _calculate_error(test_predictions, test_y)

        print(epoch, "- Train Error:", train_error, " Test Error:", test_error)

    print("Final Weights: ", weights)
    print("A: ", average)


def _run_average_epoch(weights, average, learning_rate):
    """Runs a single epoch of the Average Perceptron Algorithm"""
    for i in range(train_y.size):
        [y_example, x_example] = _get_example(i, train_y, train_data)

        if _is_error(weights, y_example, x_example):
            weights = _update_weights(weights, y_example, x_example, learning_rate)

        average = average + weights
    return [average, weights]


def _get_example(example_index, y, x):
    """Returns a single example for the given index"""
    x_example = np.empty((1, 4), dtype="float")
    for attr_index in range(4):
        x_example[0, attr_index] = x[example_index, attr_index]
    return [y[example_index], x_example]


def _is_error(weights, y_example, x_example):
    """Calculates whether the given example is correctly predicted by the given weights"""
    prediction = np.sign(x_example * np.transpose(weights))
    return y_example != prediction


def _update_weights(weights, y_example, x_example, learning_rate):
    """Updates the current weight vector by adding r * y_i * x_i"""
    return weights + learning_rate * y_example * x_example


def _calculate_error(predictions, y):
    """Calculates the error percentage based on the given predictions"""
    return np.count_nonzero(np.multiply(np.transpose(y), predictions) == -1) / len(y)


# Import
def _import_data(path, num_examples):
    """Imports the data at the given path to a csv file with the given amount of examples."""
    data = np.empty((num_examples, 4), dtype="float")
    y = np.empty((num_examples, 1), dtype="float")

    with open(path, 'r') as f:
        i = 0
        for line in f:
            example = []
            terms = line.strip().split(',')
            for j in range(len(terms)):
                if j == 4:
                    y[i] = 2 * float(terms[j]) - 1
                else:
                    example.append(float(terms[j]))
            data[i] = example
            i += 1
    return [np.asmatrix(data), np.asmatrix(y)]


if __name__ == '__main__':
    [train_data, train_y] = _import_data("./bank-note/train.csv", 872)
    [test_data, test_y] = _import_data("./bank-note/test.csv", 500)

    if sys.argv[1] == "standard":
        _run_standard_algorithm()

    elif sys.argv[1] == "voted":
        _run_voted_algorithm()

    elif sys.argv[1] == "avg":
        _run_average_algorithm()
