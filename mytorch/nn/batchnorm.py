import numpy as np

class BatchNorm1d:
    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        self.Z = Z
        self.N = Z.shape[0]  # Batch size (N)
        self.M = np.mean(Z, axis=0)  # Mean of Z (M)
        self.V = np.var(Z, axis=0)  # Variance of Z (V)

        if eval:
            # Use running averages for inference
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
        else:
            # Compute mean and variance for the batch
            self.M = np.mean(Z, axis=0, keepdims=True)
            self.V = np.var(Z, axis=0, keepdims=True)

            # Normalize the batch
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)

            # Update running mean and variance
            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

        # Scale and shift
        self.BZ = self.BW * self.NZ + self.Bb
        return self.BZ

    def backward(self, dLdBZ):
        N = dLdBZ.shape[0]

        # Gradients of scale (gamma) and shift (beta) parameters
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)

        # Intermediate partial derivatives
        dLdNZ = dLdBZ * self.BW
        dLdV = np.sum(dLdNZ * (self.Z - self.M) * -0.5 * (self.V + self.eps) ** (-1.5), axis=0, keepdims=True)
        dLdM = np.sum(dLdNZ * -1 / np.sqrt(self.V + self.eps), axis=0, keepdims=True) + dLdV * np.mean(-2 * (self.Z - self.M), axis=0, keepdims=True)

        # Gradient of the loss with respect to the inputs of the layer
        dLdZ = (dLdNZ / np.sqrt(self.V + self.eps)) + (dLdV * 2 * (self.Z - self.M) / N) + (dLdM / N)
        
        return dLdZ