import numpy as np

#######################################################
#                                                     #
#             Functions to generate data              #
#                                                     #
#######################################################


def func(X: np.ndarray) -> np.ndarray:
    """The data generating function"""
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """Add noise to the data generating function"""
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Provide training and test data for training the neural network"""
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))

    return X_train, y_train, X_test, y_test


#######################################################
#                                                     #
#             Feed-forward Neural Network             #
#                                                     #
#######################################################

def sigmoid(x):
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))


def d_sigmoid(y):
    """
    Derivative of the sigmoid activation function - input y is output of sigmoid(x)
    """
    return y * (1 - y)


def linear(x):
    """
    linear activation function
    """
    return x


def d_linear(y):
    """
    Derivative of linear activation function
    """
    return np.ones_like(y)


def mse(y, y_hat):
    """
    Mean squared error loss function
    """
    return np.mean((y - y_hat) ** 2)


def feed_forward(x, hidden_W, hidden_b, out_W, out_b):
    """
    Calculate the forward pass for input_data x
    """

    # Hidden layer calculations
    # Takes the matrix multiplication of the input and the hidden weights, then adds the hidden bias and applies the sigmoid activation function
    # Returns the activations of the hidden layer for the given input
    # This is the perceptron calculation as given in the lecture slides. g(b + sum_{i=1}^{d}(x_i * w_i)) where g is the sigmoid function :)
    hidden_activations = sigmoid(x @ hidden_W + hidden_b)
    # Stricly speaking this matrix mult. is just the dot product.

    # Output calculations
    out_linear = hidden_activations @ out_W + out_b
    y_hat = linear(out_linear)

    return y_hat, hidden_activations


def backward(y_hat, y, hidden_activations, x, out_W):
    """
    Backpropagation for the neural net in the assignment
    Parameters are those needed for the calculation

    :param y_hat: predictions from forward pass
    :param y: target data
    :param hidden_activations: Activations of hidden layer from forward pass
    :param x: input data
    :out_W: current output weights

    :returns: the derivative updates to hidden_W, hidden_b, out_W and out_b
    """

    delta_output = - 2 * (y - y_hat) * d_linear(y_hat)

    # OUTPUT LAYER (Linear Activation)

    # gradients wrt output weights
    d_L_d_ow = hidden_activations.T @ delta_output

    # gradients wrt output bias
    d_L_d_ob = np.sum(delta_output, axis=0)

    # HIDDEN LAYER (Sigmoid Activation)

    # delta term for the hidden layer
    hidden_delta = delta_output @ out_W.T * d_sigmoid(hidden_activations)

    # gradients wrt hidden weights
    d_L_d_hw = x.T @ hidden_delta

    # gradients wrt hidden bias
    d_L_d_hb = np.sum(hidden_delta, axis=0)

    return d_L_d_hw, d_L_d_hb, d_L_d_ow, d_L_d_ob


def train(x_train, y_train, neural_net, learning_rate=0.01, epochs=10, batch_size=1):
    """
    Train neural network on data
    """

    hidden_W, hidden_b, out_W, out_b = neural_net

    n_batches = int(np.ceil(x_train.shape[0]/float(batch_size)))

    for e in range(epochs):

        errors = []
        learning_rate *= 0.99
        for i in range(n_batches):

            # Forward pass
            y_hat, hidden_activations = feed_forward(x_train[batch_size*i:batch_size*(i+1)],
                                                     hidden_W, hidden_b, out_W, out_b)
            # Compute error
            error = mse(y_train[batch_size*i:batch_size*(i+1)], y_hat)
            errors.append(error)

            # Backward pass
            d_L_d_hw, d_L_d_hb, d_L_d_ow, d_L_d_ob = backward(y_hat=y_hat,
                                                              y=y_train[batch_size *
                                                                        i:batch_size*(i+1)],
                                                              hidden_activations=hidden_activations,
                                                              x=x_train[batch_size *
                                                                        i:batch_size * (i + 1)],
                                                              out_W=out_W)

            # Update parameters

            # learning_rate = eta
            # using delta_w <- eta * d_L_d_w and so on for each parameter
            hidden_W = hidden_W - learning_rate * d_L_d_hw
            hidden_b = hidden_b - learning_rate * d_L_d_hb
            out_W = out_W - learning_rate * d_L_d_ow
            out_b = out_b - learning_rate * d_L_d_ob

        print(f"Epoch {e+1}: mse {np.mean(errors):4f}", end="\n")

    return np.mean(errors), (hidden_W, hidden_b, out_W, out_b)


if __name__ == "__main__":

    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # Initialize weights for two hidden neurons
    # 2 X 2 W + 2 b
    hidden_weights = np.random.uniform(-0.5, 0.5, (2, 2))
    hidden_bias = np.random.uniform(-0.5, 0.5, 2)

    # Initialize weights for output neuron
    # 2 x 1 W + 1 b
    out_weights = np.random.uniform(-0.5, 0.5, (2, 1))
    out_bias = np.random.uniform(-0.5, 0.5, 1)

    # This is the neural net
    neural_net = (hidden_weights, hidden_bias, out_weights, out_bias)
    # Training
    train_mse, neural_net_trained = train(X_train, y_train, neural_net, learning_rate=0.2, epochs=100,
                                          batch_size=1)

    # Calculate mse on test data
    y_hat_test, _ = feed_forward(X_test, *neural_net_trained)
    test_mse = mse(y_test, y_hat_test)

    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
