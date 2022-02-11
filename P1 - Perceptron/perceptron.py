from math import e
from random import random
import numpy as np
import itertools as it

class Input():
    def __init__(self, value=random()) -> None:
        self.value = value

class Perceptron():
    def __init__(self, inputs, weights, T=random()) -> None:
        self.inputs = inputs
        self.weights = weights
        self.bias = T

    def __str__(self) -> str:
        return f"Perceptron_id: {id(self)}\nWeights: {self.weights}\nBias: {self.bias}"

    def net_input(self)-> float:
        net = []
        for i in range(len(self.inputs)):
            net.append(self.inputs[i].value*self.weights[i])
        print(f"Sum net: {sum(net)}")
        return sum(net)

    def activation(self)-> float:
        print(f"Bias: {self.bias}")
        if self.net_input() >= self.bias:
            print(f"output: 1")
            return 1
        else:
            print(f"output: 0")
            return 0

    def sigmiod(self, x):
        print(f"Activasion: {1/(1+e**-x)}")
        return 1/(1+e**-x)

