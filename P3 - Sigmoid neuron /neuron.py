from math import e

class Perceptron():
    def __init__(self, weights: list, bias: float) -> None:
        self.weights = weights
        self.b = bias

    def __str__(self) -> str:
        return f"Perceptron_id: {id(self)}\nWeights: {self.weights}\nBias: {self.b}"

    def net_input(self, inputs: list)-> float:
        """
        Bereken de net input van van de neuron en retun dit.
        """   
        net = self.b
        for i in range(len(inputs)):
            net += inputs[i]*self.weights[i]
        return net
    
    def activation(self, inputs: list)-> int:
        """
        bereken en return de activation op basis van de sigmoid function
        """           
        return self.sigmoid(self.net_input(inputs))
    
    def sigmoid(self, x: float)-> float:
        """
        Sigmoid function
        """
        return 1/(1+e**-x)

class Perceptron_layer():
    def __init__(self, perceptrons: list) -> None:
        self.perceptrons = perceptrons
    
    def activation(self, inputs: list) -> tuple:
        """
        Activate alle nodes binnen in de laag
        """
        outputs = []
        for per in self.perceptrons:
            outputs.append(per.activation(inputs))
            
        return tuple(outputs)

class Perceptro_network():
    def __init__(self, layers: list) -> None:
        self.layers = layers

    def activation(self, inputs: (int))-> tuple:
        """
        Activate alle layers in het netwerk
        """
        input = inputs
        for layer in self.layers:
            input = layer.activation(input)
        
        return input
