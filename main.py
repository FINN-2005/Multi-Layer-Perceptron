from classes import Module
import numpy as np



# Dataset: XOR 
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# Create
nn = Module(2, 4, 1)  # 2-input, 4-hidden, 1-output

# Save
nn.fit(X, Y, learning_rate=0.5, epochs=1000)
nn.save('model.weights')
del nn

# Load
nn = Module.load('model.weights')         
print('XOR {0, 0}:', nn.inference(np.array([[0],[0]])))
print('XOR {0, 1}:', nn.inference(np.array([[0],[1]])))
print('XOR {1, 0}:', nn.inference(np.array([[1],[0]])))
print('XOR {1, 1}:', nn.inference(np.array([[1],[1]])))
