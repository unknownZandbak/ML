from math import e
from random import normalvariate

class Neuron():
    def __init__(self, input_size: int) -> None:
        self.weights = []
        self.b = normalvariate(0,.2)
        self.e2 = 0
        
        for i in range(input_size):
            self.weights.append(normalvariate(0,.2))

    def __str__(self) -> str:
        return f"Neuron_id: {id(self)}\nWeights: {self.weights}\nBias: {self.b}"

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

class Neuron_layer():
    def __init__(self, input_count: int, neuron_count: int) -> None:
        self.neurons = []

        for nc in range(neuron_count):
            self.neurons.append(Neuron(input_count))
    
    def avctivate_neurons(self, inputs: list) -> tuple:
        """
        Activate alle nodes binnen in de laag
        """
        outputs = []
        for per in self.Neurons:
            outputs.append(per.activation(inputs))
            
        return tuple(outputs)

class Neuron_network():
    def __init__(self, input_size: int, layer_counts: list[int]) -> None:
        self.input_size = input_size
        self.layers = []
        
        for lci in range(len(layer_counts)):
            if lci == 0:
                self.layers.append(Neuron_layer(self.input_size, layer_counts[lci]))
            else:
                self.layers.append(Neuron_layer(layer_counts[lci-1], layer_counts[lci]))


    def feed_forward(self, inputs: (int))-> tuple:
        """
        Activate alle layers in het netwerk
        """
        input = inputs
        for layer in self.layers:
            input = layer.avctivate_neurons(input)
        
        return input

    def fit(self, train_set: list, epochs: int) -> None:
        """
        Training function to train the network
        """

        for epoch in epochs:
            print(f"\n=========Epoch {epoch+1}=========")

            for item in train_set:
                i0 = int(item[0][0])
                i1 = int(item[0][1])
