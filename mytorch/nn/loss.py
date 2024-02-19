import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = (A - Y) ** 2  
        sse = np.sum(se)  
        mse = sse / (self.N * self.C)  

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape  # TODO
        C = A.shape  # TODO
        Ones_C = np.ones(C, dtype='f')
        Ones_N = np.ones(N, dtype='f')

        shift_A = A - np.max(A, axis=1, keepdims=True)  
        exps = np.exp(shift_A)
        self.softmax = exps / np.sum(exps, axis=1, keepdims=True)

        crossentropy = -Y * np.log(self.softmax + 1e-15)  # add epsilon to prevent log(0)
        sum_crossentropy = np.sum(crossentropy)
        L = sum_crossentropy / N

        return L[0]

    def backward(self):

        N, C = self.A.shape
        dLdA = (self.softmax - self.Y) / N  # Average over batch size


        return dLdA
