import numpy as np


class SGD:

    def __init__(self, model, lr=0.1, momentum=0):

        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f")
                    for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f")
                    for i in range(self.L)]

    def step(self):

        for i in range(self.L):

            if self.mu == 0:
                self.l[i].W -= self.lr * self.l[i].dLdW
                self.l[i].b -= self.lr * self.l[i].dLdb

           
            else:
                # Update velocity for weights and compute the momentum step
                self.v_W[i] = self.mu * self.v_W[i] - self.lr * self.l[i].dLdW
                # Update velocity for biases and compute the momentum step
                self.v_b[i] = self.mu * self.v_b[i] - self.lr * self.l[i].dLdb
                # Update weights with the velocity
                self.l[i].W += self.v_W[i]
                # Update biases with the velocity
                self.l[i].b += self.v_b[i]


