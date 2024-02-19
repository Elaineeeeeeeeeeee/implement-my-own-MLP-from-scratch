import numpy as np
from scipy.special import erf

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        sigmoid_derivative = self.A * (1 - self.A)
        dLdZ = dLdA * sigmoid_derivative
        return dLdZ


class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        tanh_derivative = 1 - np.power(self.A, 2)
        dLdZ = dLdA * tanh_derivative
        return dLdZ


class ReLU:
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        relu_derivative = np.where(self.A > 0, 1, 0)
        dLdZ = dLdA * relu_derivative
        return dLdZ

class GELU:
    def forward(self, Z):
        """
        Perform the forward pass using the GELU activation function.
        """
        self.Z = Z
        self.A = 0.5 * Z * (1 + erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        """
        Perform the backward pass using the derivative of the GELU function.
        """
        # Compute the first term of the derivative
        first_term = 0.5 * (1 + erf(self.Z / np.sqrt(2)))
        
        # Compute the second term of the derivative
        second_term = (self.Z / np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.Z**2)
        
        # Combine the terms to get the full derivative
        dAdZ = first_term + second_term
        
        # Compute the gradient of the loss with respect to Z
        dLdZ = dLdA * dAdZ
        return dLdZ

class Softmax:
    def forward(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # for stability
        self.A = e_Z / np.sum(e_Z, axis=1, keepdims=True)
        return self.A
    
    def backward(self, dLdA):
        N, C = self.A.shape
        dLdZ = np.empty_like(dLdA)

        for i in range(N):
            # For each example in the batch
            J = -np.outer(self.A[i], self.A[i])  # Outer product
            J[np.arange(C), np.arange(C)] = self.A[i] * (1 - self.A[i])  # Diagonal
            dLdZ[i] = dLdA[i].dot(J)  # Chain rule

        return dLdZ


    
    def backward(self, dLdA):
        # Calculate the batch size and number of features
        N, C = self.A.shape

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros_like(self.A)

        # Fill dLdZ one data point (row) at a time
        for i in range(N):
            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i, :] = dLdA[i, :].dot(J)

        return dLdZ