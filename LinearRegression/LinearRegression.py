import sys
import numpy as np
from numpy import linalg as lin


# Stochastic Gradient Descent
def _run_stochastic_gradient_descent():
    """
    Runs stochastic gradient descent for Least Mean Squares algorithm.
    The algorithm is said to converge when the change in the current cost from the previous cost is less than 0.0001
    The learning rate is halved every 100 iterations
     """
    learning_rate = 1
    last_iter = -1

    while True:
        weights = np.zeros((101, 7), dtype="float")
        weights = np.asmatrix(weights)
        cost_string = "LEARNING RATE: " + str(learning_rate) + ":      "
        curr_cost = 0.0
        prev_cost = 100.0

        for iter in range(100):
            [data, y] = _shuffle_data(train_data, train_y)

            for example in range(53):
                prev_cost = curr_cost
                for feature in range(7):
                    h = data[example] * np.transpose(weights[iter])
                    error = y[example] - h
                    weights[iter + 1, feature] = weights[iter, feature] + learning_rate * error * data[example, feature]

                curr_cost = _compute_cost(data, y, weights[iter+1])

                cost_string += str(curr_cost) + ","
            if np.abs(curr_cost - prev_cost) < 0.0001:
                last_iter = iter + 1
                print(cost_string)
                return [weights[last_iter], learning_rate]

        print(cost_string)
        learning_rate *= 0.5


def _shuffle_data(data, y):
    """Shuffles the given data by appending y to the data, then shuffling, then returns the separated data and y."""
    combined = np.c_[data.reshape(len(data), -1), y.reshape(len(y), -1)]
    np.random.shuffle(combined)
    shuffled_data = combined[:, :data.size // len(data)].reshape(data.shape)
    shuffled_y = combined[:, data.size // len(data):].reshape(y.shape)
    return [shuffled_data, shuffled_y]


# Batch Gradient Descent
def _run_batch_gradient_descent():
    """
    Runs Batch Gradient Descent for Least Mean Squares algorithm.
    The algorithm is said to converge when ||new weight vector - prev weight vector|| is less than 0.000001.
    The learning rate is halved every 10000 iterations
    """
    learning_rate = 1
    last_iter = -1

    while True:
        weights = np.zeros((10001, 7), dtype="float")
        weights = np.asmatrix(weights)
        cost_string = "LEARNING RATE: " + str(learning_rate) + ":      "

        for iter in range(10000):
            gradient = _compute_gradient(weights[iter])
            weights[iter + 1] = weights[iter] - learning_rate * np.transpose(gradient)

            cost_string += str(_compute_cost(train_data, train_y, weights[iter])) + ","
            if lin.norm(weights[iter+1] - weights[iter]) < 0.000001:
                last_iter = iter+1
                print(cost_string)
                return [weights[last_iter], learning_rate]
        print(cost_string)
        learning_rate *= 0.5


def _compute_gradient(weights):
    """Computes the gradient of Least Mean Squares cost function w.r.t. the weight vector."""
    gradient = 0.0
    h = train_data * np.transpose(weights)
    error = train_y - h
    gradient -= np.transpose(train_data) * error
    return gradient


def _compute_cost(data, y, weights):
    """Computes the Least Mean Squares cost."""
    summed_costs = 0.0
    num_examples = int(data.size / 7)
    for example in range(num_examples):
        cost = y[example] - train_data[example]*np.transpose(weights)
        summed_costs += np.square(cost)
    return float(0.5 * summed_costs)


# Import
def _import_data(path, num_examples):
    """Imports the data at the given path to a csv file with the given amount of examples."""
    data = np.empty((num_examples, 7), dtype="float")
    y = np.empty((num_examples, 1), dtype="float")

    with open(path, 'r') as f:
        i = 0
        for line in f:
            example = []
            terms = line.strip().split(',')
            for j in range(len(terms)):
                if j == 7:
                    y[i] = float(terms[j])
                else:
                    example.append(float(terms[j]))
            data[i] = example
            i += 1
    return [np.asmatrix(data), np.asmatrix(y)]


if __name__ == '__main__':
    [train_data, train_y] = _import_data("./concrete/train.csv", 53)
    [test_data, test_y] = _import_data("./concrete/test.csv", 50)

    if sys.argv[1] == "bgd":
        [w, r] = _run_batch_gradient_descent()
        print("W: ", w)
        print("R: ", r)
        print("TEST COST: ", _compute_cost(test_data, test_y, w))
    elif sys.argv[1] == "sgd":
        [w, r] = _run_stochastic_gradient_descent()
        print("W: ", w)
        print("R: ", r)
        print("TEST COST: ", _compute_cost(test_data, test_y, w))

