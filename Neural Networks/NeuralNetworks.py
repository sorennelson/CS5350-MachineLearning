import sys
import numpy as np
import copy
from sklearn.preprocessing import normalize


def run_ann(zeros):
    activations = [sigmoid_activation, sigmoid_activation, linear_activation]
    deriv_activations = [sigmoid_activation_deriv, sigmoid_activation_deriv, linear_activation_deriv]
    # widths_values = [5, 10, 25, 50, 100]
    widths_values = [5, 10]
    print("TUNING HYPER PARAMETERS:")
    [gamma, d, _] = compute_hyperparameters(activations, deriv_activations, zeros)
    print("GAMMA:", gamma, "D:", d)

    errors = []
    print("USING HYPER PARAMETERS FOR EACH WIDTH:")
    for width in widths_values:
        widths = [5, width, width, 1]
        weights = run_sgd(gamma, d, widths, activations, deriv_activations, zeros)

        train_predictions = calculate_predictions(train_data, train_y, widths, activations, weights)
        train_error = calculate_error(train_y, train_predictions)

        test_predictions = calculate_predictions(test_data,test_y, widths, activations, weights)
        test_error = calculate_error(test_y, test_predictions)

        errors.append([width, train_error, test_error])

        print(width, " COMPLETE")

    for error in errors:
        print("Width:", error[0], ", Train Error:", error[1], ", Test Error:", error[2])


def compute_hyperparameters(activations, deriv_activations, zeros):
    gammas = [1, 0.5]
    ds = [1, 0.1]
    smallest = [0, 100.0, 100.0]

    for gamma in gammas:
        for d in ds: smallest = get_smallest_error(smallest, gamma, d, activations, deriv_activations, zeros)
        print("----------------------")
    return smallest


def get_smallest_error(smallest, gamma, d, activations, deriv_activations, zeros):
    # Computes the error for the given parameters and returns the error if it is the smallest, or the previous smallest.
    widths = [5, 5, 5, 1]
    weights = run_sgd(gamma, d, widths, activations, deriv_activations, zeros)

    predictions = calculate_predictions(train_data, train_y, widths, activations, weights)
    error = calculate_error(train_y, predictions)
    print("GAMMA:", gamma, " D:", d, " ERROR:", error)

    if error < smallest[2]: smallest = [gamma, d, error]
    return smallest


def calculate_predictions(data, y, widths, activations, weights):
    predictions = copy.deepcopy(y)
    for i in range(len(data)):
        predictions[i] = np.sign(run_forward_pass(weights, data[i], widths, activations)[-1])
    return predictions


def calculate_error(y, predictions):
    return 1 - np.count_nonzero(np.multiply(y, predictions) == 1) / len(y)


def run_sgd(initial_gamma, d, widths, activations, deriv_activations, zeros, n=872):
    weights = create_weights(widths, zeros)
    loss = []

    for epoch in range(100):
        learning_rate = update_learning_rate(initial_gamma, d, epoch)
        [y, x] = shuffle_data(train_y, train_data)
        l = 0
        for i in range(n):
            nodes = run_forward_pass(weights, x[i], widths, activations)
            prediction = np.sign(nodes[-1])

            weights_grad = run_backpropagation(weights, nodes, y[i], prediction, deriv_activations)
            weights = update_weights(weights, learning_rate, weights_grad)

            l += compute_loss(prediction, y[i])
        loss.append(l)
    # print("LOSS:", loss)
    return weights


def create_weights(widths, zeros):
    weights = []
    for level in range(len(widths) - 2):
        temp = []
        for j in range(widths[level]):
            if not zeros:
                temp.append(np.random.normal(0, 0.1, widths[level + 1] - 1).tolist())
            else:
                temp.append([0] * (widths[level + 1] - 1))
        weights.append(temp)

    temp = []
    for j in range(widths[level]):
        if not zeros:
            temp.append(np.random.normal(0, 0.1, 1).tolist())
        else:
            temp.append([1])
    weights.append(temp)

    return np.array(weights)


def shuffle_data(y, data):
    """Shuffles the given data by appending y to the data, then shuffling, then returns the separated data and y."""
    combined = np.c_[data.reshape(len(data), -1), y.reshape(len(y), -1)]
    np.random.shuffle(combined)
    shuffled_data = combined[:, :data.size // len(data)].reshape(data.shape)
    shuffled_y = combined[:, data.size // len(data):].reshape(y.shape)
    return [shuffled_y, shuffled_data]


def update_learning_rate(initial_gamma, d, epoch):
    return initial_gamma / (1.0 + epoch * (initial_gamma / d))


def update_weights(weights, learning_rate, weights_grad):
    for i in range(len(weights_grad)):
        for j in range(len(weights_grad[i])):
            for k in range(len(weights_grad[i][j])):
                if type(weights[i][j][k]) == np.matrix:
                    weights[i][j][k][0, 0] -= learning_rate * weights_grad[i][j][k][0, 0]
                else:
                    weights[i][j][k] -= learning_rate * weights_grad[i][j][k]

    return weights


def compute_loss(prediction, label):
    return np.square(prediction[0] - label[0, 0]) / 2


# Forward Pass
def run_forward_pass(weights, example, widths, activations):
    shape = []
    for i in range(len(widths)):
        shape.append(np.zeros(widths[i]))

    nodes = np.array(shape)
    nodes[0] = example

    for i in range(1, len(nodes)):
        nodes[i] = activations[i-1](widths[i], weights[i-1], nodes[i-1])

    return nodes


def linear_activation(width, weights, prev_nodes):
    curr_nodes = np.zeros(width)
    for j in range(len(curr_nodes)):

        for i in range(len(prev_nodes)):
            curr_nodes[j] += prev_nodes[i] * weights[i][j]

    return curr_nodes


def sigmoid_activation(width, weights, prev_nodes):
    prev_nodes = copy.deepcopy(prev_nodes)
    if prev_nodes.ndim > 1:
        prev_nodes = np.asarray(prev_nodes.T)
        prev_nodes = prev_nodes[:, 0]

    curr_nodes = np.zeros(width)
    curr_nodes[0] = 1

    for j in range(len(curr_nodes) - 1):
        z = 0

        for i in range(len(prev_nodes)):
            z += prev_nodes[i] * weights[i][j]

        curr_nodes[j + 1] = compute_sigmoid(z)

    return curr_nodes


def compute_sigmoid(z):
    return 1/(1+np.exp(-z))


# Backpropagation
def run_backpropagation(weights, nodes, y, prediction, activations):
    loss_deriv = prediction - y
    prev_node_derivs = [loss_deriv]
    weight_derivs = copy.deepcopy(weights)
    is_last_level = True

    for level in range(len(weights) - 1, -1, -1):
        weight_derivs[level] = compute_weight_derivs(weight_derivs[level], prev_node_derivs, nodes[level+1], nodes[level], activations[level])
        prev_node_derivs = compute_node_derivatives(weights[level], nodes[level], prev_node_derivs, is_last_level)
        is_last_level = False

    return weight_derivs


def compute_weight_derivs(weight_derivs, prev_node_derivs, prev_nodes, next_nodes, activation):
    start = 0
    if activation == sigmoid_activation_deriv: start = 1

    for i in range(len(weight_derivs)):
        for j in range(start, len(weight_derivs[i]) + start):
            if next_nodes.ndim == 2:
                next_nodes = copy.deepcopy(next_nodes)
                next_nodes = np.asarray(next_nodes.T)
                next_nodes = next_nodes[:, 0]

            weight_derivs[i][j-start] = activation(prev_node_derivs[j], next_nodes[i], prev_nodes[j])

    return weight_derivs


def linear_activation_deriv(prev_node_deriv, next_node, _):
    return prev_node_deriv[0] * next_node


def sigmoid_activation_deriv(prev_node_deriv, next_node, prev_node):
    return prev_node_deriv * next_node * prev_node * (1-prev_node)


def compute_node_derivatives(weights, curr_nodes, prev_node_derivs, is_last_level):
    curr_node_derivs = np.zeros(curr_nodes.shape)

    for i in range(len(curr_nodes)):
        product = 0

        for j in range(len(weights[i])):
            k = j
            if not is_last_level: k += 1
            product += weights[i][j] * prev_node_derivs[k]

        curr_node_derivs[i] = product

    return curr_node_derivs


def import_data(path, num_examples):
    """Imports the data at the given path to a csv file with the given amount of examples."""
    data = np.empty((num_examples, 5), dtype="float128")
    y = np.empty((num_examples, 1), dtype="float128")

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
            data[i, 1:] = example
            data[i, 0] = 1
            i += 1

    data = normalize(np.asmatrix(data), axis=0)
    return [data, np.asmatrix(y)]


def run_example():
    widths = np.array([3, 3, 3, 1])
    train_x = np.array([1., 1., 1.])
    train_y = np.array([1])
    weights = np.array([
        [[-1., 1.], [-2., 2.], [-3., 3.]],
        [[-1., 1.], [-2., 2.], [-3., 3.]],
        [[-1.], [2.], [-1.5]]
    ])
    # weights = np.array([
    #     [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3])],
    #     [np.array([-1, 1]), np.array([-2, 2]), np.array([-3, 3])],
    #     [np.array([-1]), np.array([2]), np.array([-1.5])]
    # ])
    activations = [sigmoid_activation, sigmoid_activation, linear_activation]
    nodes = run_forward_pass(weights, train_x, widths, activations)

    print("FORWARD PASS --------")
    for i in range(len(nodes)):
        print("LAYER:", i, ":", nodes[i])

    print("BACKPROPAGATION --------")
    deriv_activations = [sigmoid_activation_deriv, sigmoid_activation_deriv, linear_activation_deriv]
    weights_grad = run_backpropagation(weights, nodes, train_y, np.sign(nodes[-1]), deriv_activations)
    for level in weights_grad:
        print(level)
    print("__________")

    weights = update_weights(weights, 0.01, weights_grad)

    for level in weights:
        print(level)


def run_example_sgd():
    widths = np.array([3, 3, 3, 1])
    activations = [sigmoid_activation, sigmoid_activation, linear_activation]
    deriv_activations = [sigmoid_activation_deriv, sigmoid_activation_deriv, linear_activation_deriv]

    run_sgd(1, 1, widths, activations, deriv_activations, 1, False)


if __name__ == '__main__':
    if sys.argv[1] == "example": run_example()

    elif sys.argv[1] == "example_sgd":
        train_data = np.array([1., 1., 1.])
        train_data = train_data[np.newaxis, :]
        train_y = np.array([1])
        train_y = train_y[:, np.newaxis]

        run_example_sgd()

    elif sys.argv[1] == "ann":
        [train_data, train_y] = import_data("./bank-note/train.csv", 872)
        [test_data, test_y] = import_data("./bank-note/test.csv", 500)
        run_ann(False)

    elif sys.argv[1] == "zeros":
        [train_data, train_y] = import_data("./bank-note/train.csv", 872)
        [test_data, test_y] = import_data("./bank-note/test.csv", 500)
        run_ann(True)