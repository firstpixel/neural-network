"""
Licensed under the MIT License given below.
Copyright 2023 Daniel Lidstrom
Modified by Gil Beyruth, Copyright 2025

Code converted to python from original C version
CODE MODIFIED TO ADD CHECKPOINTING AND TO SAVE NETWORK PARAMETERS TO A BINARY FILE
ALSO ADDED FUNCTIONS TO LOAD NETWORK PARAMETERS FROM A BINARY FILE
ALSO ADDED SIGMOID PRIME FUNCTION AND VELOCITY ARRAYS TO TRAINER STRUCT
ALSO ADDED FUNCTION TO PRINT NETWORK PARAMETERS

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import numpy as np
import math
import os

# Linear congruential generator for reproducible randomness
P = 2147483647
A = 16807
current = 1

def Rand():
    global current
    current = (current * A) % P
    return current / P

# Logical operation functions
def xor_op(i, j):
    return i ^ j

def xnor_op(i, j):
    return 1 - (i ^ j)

def or_op(i, j):
    return i | j

def and_op(i, j):
    return i & j

def nor_op(i, j):
    return 1 - (i | j)

def nand_op(i, j):
    return 1 - (i & j)

# Activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prim(f):
    return f * (1.0 - f)

# The Network class (similar to the C struct)
class Network:
    def __init__(self, n_inputs, n_hidden, n_outputs, rand):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.weights_hidden = np.zeros((n_inputs, n_hidden))
        self.biases_hidden = np.zeros(n_hidden)
        self.weights_output = np.zeros((n_hidden, n_outputs))
        self.biases_output = np.zeros(n_outputs)
        self.hidden = np.zeros(n_hidden)
        self.output = np.zeros(n_outputs)
        
        # Xavier initialization for weights (for sigmoid activation)
        limit_hidden = math.sqrt(6.0 / (n_inputs + n_hidden))
        for i in range(n_inputs):
            for j in range(n_hidden):
                self.weights_hidden[i, j] = (rand() * 2.0 - 1.0) * limit_hidden
        
        limit_output = math.sqrt(6.0 / (n_hidden + n_outputs))
        for j in range(n_hidden):
            for k in range(n_outputs):
                self.weights_output[j, k] = (rand() * 2.0 - 1.0) * limit_output

    def predict(self, input_data):
        # Compute hidden layer activations
        z_hidden = np.dot(input_data, self.weights_hidden) + self.biases_hidden
        self.hidden = sigmoid(z_hidden)
        # Compute output layer activations
        z_output = np.dot(self.hidden, self.weights_output) + self.biases_output
        self.output = sigmoid(z_output)
        return self.output

    def free(self):
        # Python garbage collection handles memory freeing.
        pass

# The Trainer class holds gradients and velocities for momentum
class Trainer:
    def __init__(self, network):
        self.grad_hidden = np.zeros(network.n_hidden)
        self.grad_output = np.zeros(network.n_outputs)
        self.velocity_hidden = np.zeros((network.n_inputs, network.n_hidden))
        self.velocity_output = np.zeros((network.n_hidden, network.n_outputs))
    
    def train(self, network, input_data, target, lr, momentum):
        # Forward pass
        network.predict(input_data)
        
        # Compute output layer gradient
        for k in range(network.n_outputs):
            self.grad_output[k] = (network.output[k] - target[k]) * sigmoid_prim(network.output[k])
        
        # Compute hidden layer gradient
        for j in range(network.n_hidden):
            sum_val = 0.0
            for k in range(network.n_outputs):
                sum_val += self.grad_output[k] * network.weights_output[j, k]
            self.grad_hidden[j] = sum_val * sigmoid_prim(network.hidden[j])
        
        # Update output weights with momentum
        for j in range(network.n_hidden):
            for k in range(network.n_outputs):
                delta = lr * self.grad_output[k] * network.hidden[j]
                self.velocity_output[j, k] = momentum * self.velocity_output[j, k] + delta
                network.weights_output[j, k] -= self.velocity_output[j, k]
        
        # Update output biases
        for k in range(network.n_outputs):
            network.biases_output[k] -= lr * self.grad_output[k]
        
        # Update hidden weights with momentum
        for i in range(network.n_inputs):
            for j in range(network.n_hidden):
                delta = lr * self.grad_hidden[j] * input_data[i]
                self.velocity_hidden[i, j] = momentum * self.velocity_hidden[i, j] + delta
                network.weights_hidden[i, j] -= self.velocity_hidden[i, j]
        
        # Update hidden biases
        for j in range(network.n_hidden):
            network.biases_hidden[j] -= lr * self.grad_hidden[j]
    
    def free(self):
        # Python garbage collection handles memory freeing.
        pass

def print_network(network):
    print("Weights (Input -> Hidden):")
    for i in range(network.n_inputs):
        for j in range(network.n_hidden):
            print(f"{network.weights_hidden[i, j]:9.6f} ", end='')
        print()
    print("Biases (Hidden):")
    for j in range(network.n_hidden):
        print(f"{network.biases_hidden[j]:9.6f} ", end='')
    print("\nWeights (Hidden -> Output):")
    for j in range(network.n_hidden):
        for k in range(network.n_outputs):
            print(f"{network.weights_output[j, k]:9.6f} ", end='')
        print()
    print("Biases (Output):")
    for k in range(network.n_outputs):
        print(f"{network.biases_output[k]:9.6f} ", end='')
    print()

def network_save(network, filename):
    # Save network parameters using NumPy's savez (results in filename.npz)
    np.savez(filename,
             n_inputs=network.n_inputs,
             n_hidden=network.n_hidden,
             n_outputs=network.n_outputs,
             weights_hidden=network.weights_hidden,
             biases_hidden=network.biases_hidden,
             weights_output=network.weights_output,
             biases_output=network.biases_output)

def network_load(network, filename):
    # Expecting a file named "filename.npz"
    if not os.path.exists(filename + ".npz"):
        print(f"File {filename}.npz not found.")
        return -1
    data = np.load(filename + ".npz")
    n_inputs = int(data['n_inputs'])
    n_hidden = int(data['n_hidden'])
    n_outputs = int(data['n_outputs'])
    if n_inputs != network.n_inputs or n_hidden != network.n_hidden or n_outputs != network.n_outputs:
        print("Network dimensions mismatch!")
        return -1
    network.weights_hidden = data['weights_hidden']
    network.biases_hidden = data['biases_hidden']
    network.weights_output = data['weights_output']
    network.biases_output = data['biases_output']
    return 0

def main():
    CHECKPOINT_INTERVAL = 100000
    ITERS = 40000
    ITERS2 = ITERS + 4960000

    # Create a network with 2 inputs, 10 hidden neurons, and 6 outputs.
    network = Network(2, 10, 6, Rand)
    
    # Training parameters
    learning_rate = 0.1
    momentum = 0.9
    
    # Try to load a checkpoint
    checkpoint = network_load(network, "checkpoint.dat")
    if checkpoint == 0:
        print("Resumed from checkpoint.")
    else:
        print("No checkpoint found, starting fresh training.")
    
    trainer = Trainer(network)
    
    # Training data: four possible inputs
    inputs = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    
    # Expected outputs for the 6 logic operations:
    # Order: XOR, XNOR, OR, AND, NOR, NAND
    outputs = np.array([
        [xor_op(0, 0), xnor_op(0, 0), or_op(0, 0), and_op(0, 0), nor_op(0, 0), nand_op(0, 0)],
        [xor_op(0, 1), xnor_op(0, 1), or_op(0, 1), and_op(0, 1), nor_op(0, 1), nand_op(0, 1)],
        [xor_op(1, 0), xnor_op(1, 0), or_op(1, 0), and_op(1, 0), nor_op(1, 0), nand_op(1, 0)],
        [xor_op(1, 1), xnor_op(1, 1), or_op(1, 1), and_op(1, 1), nor_op(1, 1), nand_op(1, 1)]
    ], dtype=float)
    
    print("Initial results:")
    print("Input -> (XOR, XNOR, OR, AND, NOR, NAND)")
    for i in range(4):
        inp = inputs[i]
        network.predict(inp)
        # In the original C code the printing format slightly differs depending on checkpoint status.
        if checkpoint != 0:
            print(f"{inp[0]:.0f}, {inp[1]:.0f} = ", end='')
            for k in range(network.n_outputs):
                print(f"{network.output[k]:.3f} ", end='')
            print()
        else:
            print(f"{inp[0]:.0f},{inp[1]:.0f} = " +
                  " ".join(f"{o:.3f}" for o in network.output))
    
    # First training phase
    for i in range(ITERS):
        index = i % 4
        trainer.train(network, inputs[index], outputs[index], learning_rate, momentum)
        if i % CHECKPOINT_INTERVAL == 0:
            network_save(network, "checkpoint.dat")
    
    print(f"\nResults after {ITERS} iterations:")
    print("Input -> (XOR, XNOR, OR, AND, NOR, NAND)")
    for i in range(4):
        network.predict(inputs[i])
        print(f"{inputs[i][0]:.0f}, {inputs[i][1]:.0f} = " +
              " ".join(f"{o:.3f}" for o in network.output))
    
    # Second training phase
    for i in range(ITERS2):
        index = i % 4
        trainer.train(network, inputs[index], outputs[index], learning_rate, momentum)
        if i % CHECKPOINT_INTERVAL == 0:
            network_save(network, "checkpoint.dat")
    
    print(f"\nResults after {ITERS2} iterations:")
    print("Input -> (XOR, XNOR, OR, AND, NOR, NAND)")
    for i in range(4):
        network.predict(inputs[i])
        print(f"{inputs[i][0]:.0f}, {inputs[i][1]:.0f} = " +
              " ".join(f"{o:.3f}" for o in network.output))
    
    # Print final network parameters
    print("\nFinal network parameters:")
    print_network(network)
    
    trainer.free()
    network.free()

if __name__ == "__main__":
    main()
