import sys
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def run_logistic_regression(t="MAP"):
    variance = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    [gamma, d, _] = calculate_hyperparameters(1, t)
    print("TUNING HYPER PARAMETERS:")
    print("GAMMA:", gamma, "D:", d)
    errors = []

    if t == "MAP":
        print("USING HYPER PARAMETERS FOR EACH VARIANCE:")
        for v in variance:
            [weights, loss] = run_sgd(v, gamma, d, t)
            # plt.plot(loss)
            # plt.show()

            [train_error, test_error] = calculate_train_and_test_errors(weights)
            errors.append([v, train_error, test_error])
        for error in errors:
            print("Variance:", error[0], ", Train Error:", error[1], ", Test Error:", error[2])

    else:
        print("USING HYPER PARAMETERS:")
        [weights, loss] = run_sgd(0, gamma, d, t)
        # plt.plot(loss)
        # plt.show()

        [train_error, test_error] = calculate_train_and_test_errors(weights)
        print("Train Error:", train_error, ", Test Error:", test_error)


def calculate_train_and_test_errors(weights):
    train_predictions = np.sign(train_data.dot(weights.T))
    train_error = calculate_error(train_y, train_predictions)

    test_predictions = np.sign(test_data.dot(weights.T))
    test_error = calculate_error(test_y, test_predictions)

    return [train_error, test_error]


def calculate_hyperparameters(variance, t):
    """Finds the hyperparameters with the smallest train error"""
    gammas = [1, 0.1, 0.5, 0.01]
    ds = [1, 0.1, 0.05, 0.01, 0.005]

    smallest = [0, 100.0, 100.0]

    for gamma in gammas:
        for d in ds: smallest = get_smallest_error(smallest, variance, gamma, d, t)
        print("----------------------")
    return smallest


def get_smallest_error(smallest, variance, gamma, d, t):
    """Computes the error for the given parameters. Returns the error, if it is the smallest, or the previous smallest."""
    [weights, _] = run_sgd(variance, gamma, d, t)

    predictions = np.sign(train_data.dot(weights.T))
    error = calculate_error(train_y, predictions)
    print("GAMMA:", gamma, " D:", d, " ERROR:", error)

    if error < smallest[2]: smallest = [gamma, d, error]
    return smallest


def calculate_error(y, predictions):
    return 1 - np.count_nonzero(np.multiply(y, predictions) == 1) / len(y)


# SGD
def run_sgd(variance, initial_gamma, d, t, n=872):
    weights = np.array([[0., 0., 0., 0., 0.]], dtype="float128")
    loss = []

    for epoch in range(0, 100):
        learning_rate = update_learning_rate(initial_gamma, d, epoch)

        [y, data] = shuffle_data(train_y, train_data)

        for i in range(0, n):
            weights[0, -1] = 0.

            if t == "MAP":
                weights -= learning_rate * calculate_logistic_map_gradient(y[i], weights, data[i], variance, n)
                l = calculate_map_loss(y[i], weights, data[i], variance, n)
            else:
                weights -= learning_rate * calculate_logistic_mle_gradient(y[i], weights, data[i], n)
                l = calculate_mle_loss(y[i], weights, data[i], n)

            loss.append(l)

    return [weights, loss]


def update_learning_rate(initial_gamma, d, epoch):
    return initial_gamma / (1.0 + epoch * (initial_gamma / d))


def shuffle_data(y, data):
    """Shuffles the given data by appending y to the data, then shuffling, then returns the separated data and y."""
    combined = np.c_[data.reshape(len(data), -1), y.reshape(len(y), -1)]
    np.random.shuffle(combined)
    shuffled_data = combined[:, :data.size // len(data)].reshape(data.shape)
    shuffled_y = combined[:, data.size // len(data):].reshape(y.shape)
    return [shuffled_y, shuffled_data]


def calculate_sigmoid(z):
    return 1 / (1 + np.exp(-z))


# MAP
def calculate_logistic_map_gradient(y, w, x, variance, n=872):
    loss_grad = calculate_logistic_mle_gradient(y, w, x, n)
    regularization_grad = w/variance
    return loss_grad + regularization_grad


def calculate_map_loss(y, w, x, variance, n=872):
    z = np.dot(-(x * w.T).T, y)
    loss = np.log(1 + np.exp(z))
    regularization = np.dot(w, w.T) / variance
    return (n * loss + regularization)[0, 0]


# MLE
def calculate_logistic_mle_gradient(y, w, x, n=872):
    z = np.dot(x * w.T, y)
    sigmoid = calculate_sigmoid(z[0, 0])
    return - n * np.dot(y.T, x) * sigmoid


def calculate_mle_loss(y, w, x, n=872):
    z = np.dot(-(x * w.T).T, y)
    return (n * np.log(1 + np.exp(z)))[0, 0]


# Import
def import_data(path, num_examples):
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

    bias = np.tile([1.], (num_examples, 1))
    data = np.append(np.asmatrix(data), bias, axis=1)
    data = normalize(data, axis=0)

    return [data, np.asmatrix(y)]


def import_example():
    data = np.array([[0.5, -1, 0.3], [-1, -2, -2], [1.5, 0.2, -2.5]])
    y = np.array([[1], [-1], [1]])
    return [np.asmatrix(data), np.asmatrix(y)]


if __name__ == '__main__':
    [train_data, train_y] = import_data("./bank-note/train.csv", 872)
    [test_data, test_y] = import_data("./bank-note/test.csv", 500)

    if sys.argv[1] == "logistic_map": run_logistic_regression()
    elif sys.argv[1] == "logistic_mle": run_logistic_regression("MLE")
    elif sys.argv[1] == "example":
        [train_data, train_y] = import_example()
        run_sgd(1, 1, 1, "MAP", 3)

