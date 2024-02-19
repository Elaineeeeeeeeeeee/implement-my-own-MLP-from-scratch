import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.random.randn(out_features, in_features) * 0.01  # Small random numbers
        self.b = np.zeros((out_features, 1))  # Biases initialized to zero

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = A.shape[0]  # The batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        #self.Ones = np.ones((self.N,1))
        Z = A.dot(self.W.T) + self.b.T  # Matrix multiplication and adding bias

        return Z

    def backward(self, dLdZ):

        # Gradient of the loss with respect to the input
        dLdA = dLdZ.dot(self.W)
        # Gradient of the loss with respect to the weights
        self.dLdW = dLdZ.T.dot(self.A)
        # Gradient of the loss with respect to the biases, summed across examples
        self.dLdb = dLdZ.T.dot(np.ones((dLdZ.shape[0],1)))

        if self.debug:
            self.dLdA = dLdA  # Store for debugging purposes

        return dLdA
