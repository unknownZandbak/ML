from random import random
from numpy import float16

class Perceptron():
    def __init__(self, weights: list, bias: float) -> None:
        self.weights = weights
        self.b = bias
        self.e2 = 0

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

    def update(self, D, Y, X, eta)-> None:
        # bereken de error
        e = D - Y
        # ga nu voor alle inputs, de verandering van de weight berekenen en pas de weight aan.
        for input in range(len(X)):
            deltaW = eta*e*X[input]
            self.weights[input] = self.weights[input] + deltaW
        # bereken de verandering van de bias en pas de bias aan.
        deltaB = eta*e
        self.b = self.b + deltaB
        # tel mijn sqeared error op bij de vorige voor later om de MSE te berekennen.
        self.e2 += e**2
        
    def loss(self, n):
        MSE = self.e2 / n
        self.e2 = 0
        return MSE
