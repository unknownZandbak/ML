from random import random

class Perceptron():
    def __init__(self, weights, bias=random()) -> None:
        self.weights = weights
        self.b = bias

    def __str__(self) -> str:
        return f"Perceptron_id: {id(self)}\nWeights: {self.weights}\nBias: {self.b}"

    def net_input(self, inputs: list)-> float:
        # De begin waarde van de net input is de bias.
        net = self.b
        for i in range(len(inputs)):
            net += inputs[i]*self.weights[i]
        return net

    def activation(self, inputs: list)-> int:
        # hier geven we een activation terug op basis van de step-up function.
        if self.net_input(inputs) >= 0:
            return 1
        else:
            return 0

class Perceptron_layer():
    def __init__(self, perceptrons: list) -> None:
        self.perceptrons = perceptrons
    
    def activation(self, inputs: list) -> tuple:
        outputs = []
        for per in self.perceptrons:
            outputs.append(per.activation(inputs))
            
        return tuple(outputs)

class Perceptro_network():
    def __init__(self, layers: list) -> None:
        self.layers = layers

    def activation(self, inputs: (int))-> tuple:
        input = inputs
        for layer in self.layers:
            input = layer.activation(input)
        
        return input
