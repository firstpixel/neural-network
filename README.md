# Neural Network with Checkpoint

This project is a Python implementation of a feed-forward neural network originally written in C. It demonstrates:

- A simple neural network architecture with one hidden layer.
- Backpropagation training with momentum.
- Xavier initialization of weights.
- Checkpointing: saving and loading network parameters from a binary file.
- Custom logic for handling logical operations (e.g., XOR, XNOR, OR, AND, NOR, NAND).

## Overview

The code includes:

- A **linear congruential generator** for reproducible randomness.
- **Logical operation functions** to generate target outputs.
- **Activation functions**: the sigmoid and its derivative.
- A **Network** class that defines the neural network structure and performs feed-forward predictions.
- A **Trainer** class that implements backpropagation with momentum to update the network parameters.
- Utility functions for **printing**, **saving**, and **loading** the network parameters.

The network is trained using two phases and periodically saves checkpoints to allow resuming training from a saved state.

## Features

- **Feed-Forward Neural Network:** Configurable number of inputs, hidden neurons, and outputs.
- **Backpropagation with Momentum:** Improves training efficiency.
- **Xavier Initialization:** Helps speed up convergence during training.
- **Checkpointing:** Automatically saves network parameters, allowing training to be resumed.
- **Logical Operation Tasks:** Demonstrates training on logical functions like XOR, XNOR, OR, AND, NOR, and NAND.

## Requirements

- Python 3.x
- [NumPy](https://numpy.org/)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/firstpixel/neural-network.git
   cd neural-network
    ```
Install dependencies:

 ```bash
pip install numpy
 ```
Usage
Run the main script to start training the neural network:

 ```bash
python main.py
 ```
### The script will:

Initialize the network with 2 inputs, 10 hidden neurons, and 6 outputs.
Attempt to load a checkpoint (if available).
Train the network in two phases, saving checkpoints at regular intervals.
Print initial and final results, along with the final network parameters.
Code Structure
Rand() Function:
Implements a linear congruential generator for reproducible randomness.

### Logical Operation Functions:
Includes xor_op, xnor_op, or_op, and_op, nor_op, and nand_op.

### Activation Functions:
sigmoid and sigmoid_prim (its derivative).

### Network Class:
Sets up the neural network, performs Xavier initialization, and handles feed-forward prediction.

### Trainer Class:
Computes gradients and updates weights using backpropagation with momentum.

## Utility Functions:

print_network(): Prints current network parameters.
network_save(): Saves network parameters to a binary file using NumPy.
network_load(): Loads network parameters from a binary file.
main() Function:
Coordinates the training process, manages checkpointing, and prints results.


## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests with improvements or additional features.
