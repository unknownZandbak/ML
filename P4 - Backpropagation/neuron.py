from math import e
from random import normalvariate

class Neuron():
    def __init__(self, input_size: int) -> None:
        """
        initialize a neuron.
        Args:
            input_size (int): amount of inputs the neuron has
        """
        self.inputs = []
        self.weights = []
        self.b = normalvariate(0,.2)
        self.delta_weights = []
        self.delta_bias = 0
        self.out = 0
        self.e2 = 0
        
        for i in range(input_size):
            self.weights.append(normalvariate(0,.2))

    def __str__(self) -> str:
        return f"Neuron_id: {id(self)}\nWeights: {self.weights}\nBias: {self.b}"

    def net_input(self, inputs: list)-> float:
        """
        Bereken de net input van de neuron en return dit.
        """   
        net = self.b
        self.inputs = inputs
        for i in range(len(inputs)):
            net += inputs[i]*self.weights[i]
        return net
    
    def activation(self, inputs: list)-> int:
        """
        bereken en return de activation op basis van de sigmoid function
        """           
        self.out = self.sigmoid(self.net_input(inputs))
        return self.out
    
    def sigmoid(self, x: float)-> float:
        """
        Sigmoid function.
        """
        return 1/(1+e**-x)

class Neuron_layer():
    def __init__(self, input_count: int, neuron_count: int) -> None:
        """
        initialize a neuron layer.

        Args:
            input_count (int): amount of inputs neuron of this layer wil have
            neuron_count (int):amount of neuron for this layers
        """
        self.neurons = []

        # init the given amount of neurons,
        # and pass along the amount of inputs the neuron has.
        for nc in range(neuron_count):
            self.neurons.append(Neuron(input_count))
    
    def activate_neurons(self, inputs: list) -> list:
        """
        Activate alle nodes binnen in de laag
        """
        outputs = []
        for per in self.neurons:
            outputs.append(per.activation(inputs))
            
        return outputs

    def fit_output_layer(self, target: int, eta: float) -> None:
        """
        Calculate all the errors Weights adn bias's for teh neurons in the output layer.

        Args:
            target (int): Target value of what we want the output to be.
            eta (float): Constant leraning rate.
        """
        #  then append those to a list for later use
        for neuron in self.neurons:
            # calculate the error of set neuron
            target = target if type(target) != list else target[self.neurons.index(neuron)]
            err = neuron.out*(1-neuron.out)*-(target - neuron.out)
            neuron.e2 = err
            
            # calculate the new weights for set neuron
            delta_w = []
            for w in range(len(neuron.weights)):
                delta_w.append(eta*neuron.inputs[w]*err)
                neuron.delta_weights.append(delta_w)    

            # calculate the new bias for set neuron.
            delta_b = eta*err
            neuron.delta_bias = delta_b

    def fit_hidden_layer(self, target: int, perv_layer, eta: float) -> None:
        """
        Calculate all the errors Weights adn bias's for teh neurons in this hidden layer.

        Args:
            target (int): Target value of what we want the output to be.
            eta (float): Constant leraning rate.
        """
        #  then append those to a list for later use
        for neuron in self.neurons:

            # calculate the error of set neuron
            som = 0
            for prev_neuron in perv_layer.neurons:
                som =+ prev_neuron.weights[self.neurons.index(neuron)]*prev_neuron.e2
            err = neuron.out*(1-neuron.out)*(som)
            neuron.e2 = err
            
            # calculate the new weights for set neuron
            delta_w = []
            for w in range(len(neuron.weights)):
                delta_w.append(eta*neuron.inputs[w]*err)
                neuron.delta_weights.append(delta_w)      

            # calculate the new bias for set neuron.
            delta_b = eta*err
            neuron.delta_bias = delta_b
        


class Neuron_network():
    def __init__(self, input_size: int, layer_counts: list[int]) -> None:
        self.input_size = input_size
        self.layers = []

        # init the layers with the given amount of neurons
        for lci in range(len(layer_counts)):
            if lci == 0:
                self.layers.append(Neuron_layer(self.input_size, layer_counts[lci]))
            else:
                self.layers.append(Neuron_layer(layer_counts[lci-1], layer_counts[lci]))

    def predict(self, inputs: list)-> list:
        """
        Activate all layers in the network.
        """
        for layer in self.layers:
            inputs = layer.activate_neurons(inputs)
        return inputs

    def train(self, data: list, targets: list, epochs: int, eta=.5) -> None:
        """
        Train the network with the given training data.

        Args:
            data (list): data set used for training.
            targets (list): targets or the data set
            epochs (int): amount of epchos.
            eta (0.5): Constant learning rate.
        """
        for epoch in range(epochs):
            print(f"\n=========Epoch {epoch+1}=========")

            # quick check to see if the data and target list are the same size.
            if len(data) == len(targets):

                for index in range(len(data)):

                    # seprate the inputs and target values
                    train_input = data[index]
                    target = targets[index]

                    Y = self.predict(train_input)
                    print(f"Input: {train_input} || gekregen Output: {Y} || Correcte Output: {target}")

                    # go trough the layers to establish what the new values should be
                    for layer in range(1,len(self.layers)+1):
                        if layer != 1:
                            self.layers[-layer].fit_hidden_layer(target, self.layers[-(layer-1)], eta)
                        else:
                            self.layers[-layer].fit_output_layer(target, eta)
            else:
                raise ValueError("Data set and Targets are not the same size")
