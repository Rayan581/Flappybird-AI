import random
import math
import json
import os
from datetime import datetime


class NeuralNetwork:
    class Layer:
        def __init__(self, input_size, output_size, activation='sigmoid'):
            self.input_size = input_size
            self.output_size = output_size
            self.activation = activation
            self.weights = []
            self.biases = []

            # Xavier initialization for better starting weights
            limit = math.sqrt(6.0 / (input_size + output_size))
            for _ in range(self.output_size):
                self.weights.append([random.uniform(-limit, limit)
                                    for _ in range(self.input_size)])
                self.biases.append(random.uniform(-limit, limit))

        def forward(self, inputs):
            output = []
            for weights, bias in zip(self.weights, self.biases):
                activation = sum(w * i for w, i in zip(weights, inputs)) + bias

                if self.activation == 'sigmoid':
                    activation = 1 / \
                        (1 + math.exp(-max(-500, min(500, activation))))
                elif self.activation == 'tanh':
                    activation = math.tanh(activation)
                elif self.activation == 'relu':
                    activation = max(0, activation)
                elif self.activation == 'leaky_relu':
                    activation = activation if activation > 0 else 0.01 * activation

                output.append(activation)
            return output

        def copy(self):
            new_layer = NeuralNetwork.Layer(
                self.input_size, self.output_size, self.activation)
            new_layer.weights = [w[:] for w in self.weights]
            new_layer.biases = self.biases[:]
            return new_layer

        def set_weights(self, weights, biases):
            self.weights = [w[:] for w in weights]
            self.biases = biases[:]

        def to_dict(self):
            return {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'activation': self.activation,
                'weights': self.weights,
                'biases': self.biases
            }

        @classmethod
        def from_dict(cls, data):
            layer = cls(data['input_size'],
                        data['output_size'], data['activation'])
            layer.weights = data['weights']
            layer.biases = data['biases']
            return layer

    def __init__(self, layer_sizes, activations=None):
        """
        layer_sizes: list like [input_size, hidden1_size, ..., output_size]
        activations: list of activation functions for each layer (except input)
        """
        self.layers = []
        if activations is None:
            activations = ['leaky_relu'] * (len(layer_sizes) - 2) + ['sigmoid']

        for i in range(len(layer_sizes) - 1):
            self.layers.append(NeuralNetwork.Layer(
                layer_sizes[i], layer_sizes[i + 1], activations[i]))

    def forward(self, inputs):
        """Passes inputs through all layers and returns final output."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def copy(self):
        """Creates a deep copy of the entire network."""
        layer_sizes = [layer.input_size for layer in self.layers] + \
            [self.layers[-1].output_size]
        activations = [layer.activation for layer in self.layers]
        new_net = NeuralNetwork(layer_sizes, activations)
        new_net.layers = [layer.copy() for layer in self.layers]
        return new_net

    def set_weights(self, weights_list, biases_list):
        """Sets the weights and biases for all layers."""
        for layer, w, b in zip(self.layers, weights_list, biases_list):
            layer.set_weights(w, b)

    def get_weights(self):
        """Returns a tuple (weights_list, biases_list) for all layers."""
        weights_list = [layer.weights[:] for layer in self.layers]
        biases_list = [layer.biases[:] for layer in self.layers]
        return weights_list, biases_list

    def save_to_file(self, filename=None, metadata=None):
        """Save the neural network to a JSON file."""
        if filename is None:
            filename = f"best_bird_brain.json"

        # Ensure saves directory exists
        os.makedirs("saves", exist_ok=True)
        filepath = os.path.join("saves", filename)

        data = {
            'metadata': metadata or {},
            'layers': [layer.to_dict() for layer in self.layers],
            'architecture': [layer.input_size for layer in self.layers] + [self.layers[-1].output_size],
            'activations': [layer.activation for layer in self.layers],
            'saved_at': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Neural network saved to: {filepath}")
        return filepath

    @classmethod
    def load_from_file(cls, filepath):
        """Load a neural network from a JSON file."""
        print(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create network with same architecture
        network = cls(data['architecture'], data['activations'])
        network.layers = [cls.Layer.from_dict(
            layer_data) for layer_data in data['layers']]

        return network, data.get('metadata', {})
