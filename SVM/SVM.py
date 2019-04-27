import sys
import numpy as np
from scipy.optimize import minimize


# Dual
def run_dual(kernel, g=False):
    cs = [100/873, 500/873, 700/873]
    gammas = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]
    # gammas = [2]
    guess = np.random.rand(872)
    con = [{'type': 'eq', 'fun': constraint}]

    for c in cs:
        if g:
            for g in gammas:
                solution = minimize(dual_objective, guess, args=[kernel, g], method='SLSQP', bounds=bounds(c),
                                    constraints=con)
                compute_and_print_errors(solution, c, g, kernel)
        else:
            solution = minimize(dual_objective, guess, args=[kernel, 0], method='SLSQP', bounds=bounds(c), constraints=con)
            compute_and_print_errors(solution, c, 0, kernel)


def bounds(c):
    return [(0., c)] * 872


def constraint(alpha):
    return alpha.dot(train_y)[0, 0]


def dual_objective(alpha, args):
    kernel = args[0]
    g = args[1]
    y = train_y
    x = train_data
    k = kernel(x, x, g)

    yky = y.T * k * y
    left_summation = alpha.T.dot(np.dot(yky[0, 0], alpha))
    return 1/2 * left_summation - np.sum(alpha)


def linear_kernel(x, z, _=0):
    return x * z.T


def gaussian_kernel(x, z, g):
    return np.exp(-np.sum(np.square(x - z)) / g)


def compute_and_print_errors(solution, c, g, kernel):
    alpha = clean_alpha(solution.x, c)

    support_indices = np.where(0 < alpha)[0]  # index of support vectors
    [weights, bias, train_error, test_error] = compute_dual_train_and_test_errors(alpha, kernel, g)

    print("SUPPORT VECTOR COUNT:", len(support_indices))
    print("WEIGHTS:", weights)
    print("BIAS:", np.average(bias))
    print("TRAIN ERROR:", train_error)
    # print("TEST ERROR:", test_error)
    print("G:", g)
    print("C:", c)
    print("_________")


def clean_alpha(alpha, c):
    # Sets values close to 0 to be zero, and values close to c to c.
    # Uses relative tolerance of 1e-05, absolute tolerance of 1e-08
    alpha[np.isclose(alpha, 0)] = 0
    alpha[np.isclose(alpha, c)] = c
    return alpha


def compute_dual_train_and_test_errors(alpha, kernel, g):
    # Calculates the weights and bias terms from the given alpha values,
    # then computes train and test errors using the given kernel
    weights = compute_dual_weights(alpha)
    bias = compute_dual_bias(alpha, train_y, train_data, kernel, g)

    train_error = compute_dual_error(alpha, train_y, train_data, bias, kernel, g)
    # test_error = compute_dual_error(alpha, test_y, test_data, bias, kernel)
    return [weights, bias, train_error, 0]


def compute_dual_weights(alpha):
    return np.sum((alpha * train_y)[0, 0] * train_data, axis=0)


def compute_dual_bias(alpha, y, x, kernel, g):
    return y - np.sum((alpha * y)[0, 0] * kernel(x, x, g))


def compute_dual_error(alpha, y, x, b, kernel, g):
    # Computes the predictions for the given weights/data. Then calculates the error for the predictions.
    predictions = compute_dual_predictions(alpha, x, y, b, kernel, g).T
    return compute_error(y, predictions)


def compute_dual_predictions(alpha, x, y, b, kernel, g):
    return np.sign(np.sum((alpha * y)[0, 0] * kernel(train_data, x, g) + b, axis=1))


# Primal
def run_primal_svm(schedule=1):
    # Calculates the hyperparameters, then computes the train and test errors for every value of c.
    cs = [1 / 873, 10 / 873, 50 / 873, 100 / 873, 300 / 873, 500 / 873, 700 / 873]
    train_errors = []
    test_errors = []

    hyperparameters = compute_primal_hyper_parameters(schedule)
    gamma = hyperparameters[0]
    d = hyperparameters[1]

    print("GAMMA: ", gamma)
    if schedule == 1: print("d: ", d)

    for c in cs:
        [train_error, test_error] = compute_primal_train_and_test_errors(c, gamma, d)
        train_errors.append([c, train_error])
        test_errors.append([c, test_error])

    for i in range(len(train_errors)):
        print("C:", cs[i], " Train Error:", train_errors[i][1], " Test Error:", test_errors[i][1])


def compute_primal_train_and_test_errors(c, gamma, d):
    # Calculates the weights by running SGD, then computes train and test errors for the given parameters
    [weights, _] = run_sgd(c, gamma, d)
    weights = weights[0, :weights.size - 1]

    train_error = compute_primal_error(weights, train_y, train_data)
    test_error = compute_primal_error(weights, test_y, test_data)
    return [train_error, test_error]


def compute_primal_hyper_parameters(schedule):
    # Computes the best hyper parameters for c = 700 / 873
    gammas = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
    ds = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    smallest = [0, 100.0, 100.0]
    c = 700 / 873

    for gamma in gammas:
        if schedule == 1:
            for d in ds: smallest = get_smallest_error(smallest, c, gamma, d)
        else:
            smallest = get_smallest_error(smallest, c, gamma)

    return smallest


def get_smallest_error(smallest, c, gamma, d=0):
    # Computes the error for the given parameters and returns the error if it is the smallest, or the previous smallest.
    [weights, _] = run_sgd(c, gamma, d)
    weights = weights[0, :weights.size - 1]
    error = compute_primal_error(weights, train_y, train_data)

    if error < smallest[2]: smallest = [gamma, d, error]
    return smallest


def compute_primal_error(weights, y, x):
    # Computes the predictions for the given weights/data. Then calculates the error for the predictions.
    predictions = compute_primal_predictions(weights, x)
    return compute_error(y, predictions)


def compute_primal_predictions(w, x):
    return np.sign(w * np.transpose(x))


# SGD
def run_sgd(c, g_not, d, n=872):
    weights = np.array([[0., 0., 0., 0., 0.]])
    loss = []

    for epoch in range(0, 100):
        gamma = update_gamma(epoch, g_not, d)
        [temp_train_y, temp_train_set] = _shuffle_data(train_y, train_data)

        for i in range(0, n):
            x = np.append(temp_train_set[i], [[1]], axis=1)

            if is_incorrectly_classified(temp_train_y[i], weights, x):
                weights = update_incorrect_example(weights, gamma, c, temp_train_y[i], x, n)
            else:
                weights[0][:weights.size-1] = update_correct_example(weights, gamma)

            l = compute_loss(c, temp_train_y, weights, temp_train_set, n)
            loss.append(l[0, 0])
            # print("LOSS: ", l[0, 0])
            # If convergent, then return w
    return [weights, loss]


def update_gamma(t, g_not, d):
    if d != 0: return g_not / (1.0 + (g_not / d) * t)
    else: return g_not / (1 + t)


def _shuffle_data(y, data):
    """Shuffles the given data by appending y to the data, then shuffling, then returns the separated data and y."""
    combined = np.c_[data.reshape(len(data), -1), y.reshape(len(y), -1)]
    np.random.shuffle(combined)
    shuffled_data = combined[:, :data.size // len(data)].reshape(data.shape)
    shuffled_y = combined[:, data.size // len(data):].reshape(y.shape)
    return [shuffled_y, shuffled_data]


def is_incorrectly_classified(y_i, w, x_i):
    return y_i * np.dot(x_i, np.transpose(w)) <= 1


def update_incorrect_example(w, gamma, c, y_i, x_i, n=872):
    w[0, w.size - 1] = 0
    temp = (1-gamma) * w + gamma * c * n * y_i * x_i
    return temp[0]


def update_correct_example(w, gamma):
    return (1-gamma) * w[:w.size - 1]


def compute_loss(c, y, w, x, n=872):
    x = np.append(x, np.ones((n, 1)), axis=1)
    hinge = max(0, 1 - np.transpose(y) * np.dot(x, np.transpose(w)))

    w = w[0, :w.size - 1]
    regularization = 1/2 * w.dot(w.T)
    return regularization + c * hinge


# Primal and Dual
def compute_error(y, predictions):
    return 1 - np.count_nonzero(np.multiply(y.T, predictions) == 1) / len(y)


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
    return [np.asmatrix(data), np.asmatrix(y)]


def run_primal_example():
    run_sgd(1 / 873, 0.01, d=0.01, n=3)


def import_example():
    data = np.array([[0.5, -1, 0.3], [-1, -2, -2], [1.5, 0.2, -2.5]])
    y = np.array([[1], [-1], [1]])
    return [np.asmatrix(data),  np.asmatrix(y)]


if __name__ == '__main__':
    [train_data, train_y] = import_data("./bank-note/train.csv", 872)
    [test_data, test_y] = import_data("./bank-note/test.csv", 500)

    if sys.argv[1] == "primal":
        run_primal_svm()
    elif sys.argv[1] == "primal2":
        run_primal_svm(2)
    elif sys.argv[1] == "example":
        [train_data, train_y] = import_example()
        run_primal_example()
    elif sys.argv[1] == "linear_dual":
        run_dual(linear_kernel)
    elif sys.argv[1] == "gaussian_dual":
        run_dual(gaussian_kernel, g=True)
