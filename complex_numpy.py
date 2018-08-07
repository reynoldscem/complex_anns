from autograd.scipy.misc import logsumexp
import autograd.numpy as np
from autograd import grad


LR = 0.003
MAX_ITER = 7500
BATCH_SIZE = 64


class ComplexANN():
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        assert n_hidden >= 1

        self.input_size = input_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.parameters = []
        self.parameters.append(
            self.complex_init(input_size, hidden_size)
        )

        for _ in range(2, self.n_hidden):
            self.parameters.append(
                self.complex_init(hidden_size, hidden_size)
            )

        self.parameters.append(
            self.complex_init(hidden_size, output_size)
        )

        self.gradient_function = None
        self.lr = None

    @classmethod
    def complex_init(cls, incoming, outgoing):
        # Similar to glorot initialisation.
        real_rand = (np.random.randn(incoming, outgoing) - 0.5)
        real_rand = real_rand * 1 / (incoming + outgoing)
        imag_rand = (np.random.randn(incoming, outgoing) - 0.5)
        imag_rand = imag_rand * 1 / (incoming + outgoing)

        bias = np.zeros(outgoing) + 1j * np.zeros(outgoing)

        return (real_rand + 1j * imag_rand, bias)

    @classmethod
    def forward(cls, parameters, incoming):
        for weight, bias in parameters:
            outgoing = np.dot(incoming, weight) + bias
            incoming = cls.nonlinearity(outgoing)

        outgoing = np.real(outgoing)

        return outgoing - logsumexp(outgoing, axis=1, keepdims=True)

    @staticmethod
    def nonlinearity(value):
        return np.tanh(np.real(value)) + np.tanh(np.imag(value))

    @classmethod
    def neg_log_likelihood(cls, parameters, inputs, targets):
        return -np.sum(cls.forward(parameters, inputs) * targets)

    @classmethod
    def accuracy(cls, parameters, inputs, targets):
        target_numeric = np.argmax(targets, axis=1)
        output = cls.forward(parameters, inputs)
        prediction_numeric = np.argmax(output, axis=1)

        return np.mean(prediction_numeric == target_numeric)

    def initialise_optimiser(
            self, gradient_function, lr=0.01, lr_drop_iters=[]):
        self.gradient_function = gradient_function
        self.lr_drop_iters = lr_drop_iters
        self.lr = lr

    def step(self, i):
        if self.gradient_function is None or self.lr is None:
            raise RuntimeError('Optimiser not initialised!')

        if i in self.lr_drop_iters:
            self.lr /= 2
            print('Dropping lr to {}'.format(self.lr))

        updates = self.gradient_function(self.parameters, i)
        param_enumerator = enumerate(zip(self.parameters, updates))
        for index, (params, update) in param_enumerator:
            updated_parameters = tuple(
                params[i] - self.lr * update[i]
                for i in (0, 1)
            )
            self.parameters[index] = updated_parameters


def one_hot(integer_labels):
    from sklearn.preprocessing import OneHotEncoder
    integer_labels = np.array(integer_labels)
    assert integer_labels.ndim == 1
    encoder = OneHotEncoder(sparse=False)

    # Add an extra singleton dim at the end.
    return encoder.fit_transform(integer_labels.reshape(-1, 1))


def _batch_indices(num_batches, iteration):
    idx = iteration % num_batches
    return slice(idx * BATCH_SIZE, (idx+1) * BATCH_SIZE)


def plot_stats(losses, accuracies):
    from matplotlib import pyplot as plt
    plt.plot(losses)
    plt.show()

    plt.figure()
    plt.plot(accuracies)
    plt.show()


def main():
    from sklearn.datasets import load_digits
    from functools import partial

    # 64 Input features.
    # 2 Hidden Layers.
    # 300 Nodes per hidden layer.
    # 10 Output units.
    my_nn = ComplexANN(64, 2, 100, 10)

    data = load_digits()
    X = data.data
    y = one_hot(data.target)

    num_batches = int(np.ceil(len(X) / BATCH_SIZE))
    batch_indices = partial(_batch_indices, num_batches)

    def objective(parameters, batch_index):
        indices = batch_indices(batch_index)
        return ComplexANN.neg_log_likelihood(
            parameters, X[indices], y[indices]
        )
    objective_grad = grad(objective)

    def accuracy(parameters):
        return ComplexANN.accuracy(parameters, X, y)

    lr_drop_iters = [MAX_ITER // 2, int(MAX_ITER * 0.75)]
    my_nn.initialise_optimiser(objective_grad, LR, lr_drop_iters)

    losses, accuracies = [], []
    for i in range(MAX_ITER):
        my_nn.step(i)

        losses.append(objective(my_nn.parameters, i))

        if i % 100 == 0:
            acc = 100 * accuracy(my_nn.parameters)
            print(
                'Iteration {:d},\tTraining accuracy {:03.3f}%'
                ''.format(i, acc)
            )
            accuracies.append(acc)

    plot_stats(losses, accuracies)


if __name__ == '__main__':
    main()
