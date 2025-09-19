import pickle
import numpy as np


np.random.seed(1337)

def sigmoid(x:float) -> float:
    return (1 / (1 + (np.e**(-x))))

def d_sigmoid(x:float) -> float:
    y = sigmoid(x)
    return y * (1 - y)

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def d_cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))




class Layer:
    def __init__(self, n, b):
        self.weights = np.random.randn(b, n)
        self.biases = np.zeros((b, 1))

    def forward(self, x_inputs: np.ndarray):
        self.input = x_inputs
        self.z = self.weights @ x_inputs + self.biases
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, dC_dA: np.ndarray, learning_rate: float):
        dA_dZ = d_sigmoid(self.z)               # σ'(z)
        dC_dZ = dC_dA * dA_dZ                   # δ = dC/dA * σ'(z)
        
        dC_dW = dC_dZ @ self.input.T            # dC/dW
        dC_dB = dC_dZ                           # dC/dB
        dC_dA_prev = self.weights.T @ dC_dZ     # gradient to pass backward

        self.weights -= learning_rate * dC_dW
        self.biases -= learning_rate * np.sum(dC_dB, axis=1, keepdims=True)
        return dC_dA_prev
    






class Module:
    def __init__(self, *args):
        if len(args) < 2:
            raise Exception('Need at least input and output layer sizes.')
        self.layers = []
        for i in range(len(args) - 1):
            self.layers.append(Layer(n=args[i], b=args[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dC_dA, learning_rate):
        for layer in reversed(self.layers):
            dC_dA = layer.backward(dC_dA, learning_rate)

    def inference(self, x):
        return self.forward(x)

    def train(self, x, y_true, learning_rate):
        y_pred = self.forward(x)
        loss = cross_entropy_loss(y_true, y_pred)
        dC_dA = d_cross_entropy(y_true, y_pred)
        self.backward(dC_dA, learning_rate)
        return y_pred, loss

    def fit(self, X, Y, learning_rate=0.1, epochs=1000, batch_size=128, verbose=True):
        for epoch in range(epochs):
            # Shuffle the data at the start of each epoch
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            total_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]

                # Vectorized batch processing
                Y_pred = self.forward(X_batch.T)
                loss = cross_entropy_loss(Y_batch.T, Y_pred)
                dC_dA = d_cross_entropy(Y_batch.T, Y_pred)
                self.backward(dC_dA, learning_rate)
                total_loss += loss

            if verbose and epoch % 10 == 0:  # Reduce verbosity to every 10 epochs
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch}: Loss = {avg_loss.item():.6f}")

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
