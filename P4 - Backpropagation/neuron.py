from math import e
from tokenize import String
from numpy import append, argmax
from tqdm import trange
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
        self.b = normalvariate(0,.1)
        self.delta_weights = []
        self.delta_bias = 0
        self.out = 0
        self.e = 0
        
        for i in range(input_size):
            self.weights.append(normalvariate(0,.1))

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
            tmp_target = target if type(target) != list else target[self.neurons.index(neuron)]
            err = neuron.out*(1-neuron.out)*-(tmp_target - neuron.out)
            neuron.e = err
            
            # calculate the new weights for set neuron
            for w in range(len(neuron.weights)):
                neuron.delta_weights.append(eta*neuron.inputs[w]*err)   

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
                som =+ prev_neuron.weights[self.neurons.index(neuron)]*prev_neuron.e
            err = neuron.out*(1-neuron.out)*(som)
            neuron.e = err
            
            # calculate the new weights for set neuron
            for w in range(len(neuron.weights)):
                neuron.delta_weights.append(eta*neuron.inputs[w]*err)     

            # calculate the new bias for set neuron.
            delta_b = eta*err
            neuron.delta_bias = delta_b
    
    def apply_values(self) -> None:
        """
        With this method we will apply all the newly calculated values for the nerons in this layer.
        """
        for neuron in self.neurons:
            # apply the weights
            for wi in range(len(neuron.weights)):
                neuron.weights[wi] -= neuron.delta_weights[wi]
            neuron.delta_weights = []
            # apply the bias
            neuron.b -= neuron.delta_bias
            neuron.delta_bias = 0



class Neuron_network():
    def __init__(self, input_size: int, layer_counts: list[int]) -> None:
        self.input_size = input_size
        self.layers = []
        self.accuracy_l = []
        self.MSE_l = []

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

    def MSE_acc(self, Y, target) -> None:
        # use argmax to collect accuracy || Assumes there is a 1 in the target list for classifying
        mxi = argmax(Y)
        if 1 in target:
            if mxi == target.index(1): self.accuracy_l.append(1)
            else: self.accuracy_l.append(0)

        # collect SE of this network model.
        for layer in self.layers:
            for neuron in layer.neurons:
                self.MSE_l.append(neuron.e**2)

    def train(self, X: list, y: list, epochs: int, eta=.5) -> None:
        """
        Train the network with the given training data.

        Args:
            X (list): data set used for training.
            y (list): targets or the data set.
            epochs (int): amount of epochs set to train the network for.
            eta (0.5): Constant learning rate default=0.5.

        Raises:
            ValueError: Data set and Targets are not the same size
        """
        for epoch in trange(epochs, desc="Progress", unit=" epoch"):

            # quick check to see if the data and target list are the same size.
            if len(X) == len(y):

                for index in range(len(X)):

                    # seprate the inputs and target values
                    train_input = X[index]
                    target = y[index]

                    Y = self.predict(train_input)
                    # print(f"Input: {train_input} || gekregen Output: {Y} || Correcte Output: {target}")

                    # go trough the layers to calculate all the new values.
                    for layer in range(1,len(self.layers)+1):
                        if layer != 1:
                            self.layers[-layer].fit_hidden_layer(target, self.layers[-(layer-1)], eta)
                        else:
                            self.layers[-layer].fit_output_layer(target, eta)

                    # once everything has been calculated we can apply the new values.
                    for layer in self.layers:
                        layer.apply_values()
                    
                    self.MSE_acc(Y, target)
        
            else:
                raise ValueError("Data set and Targets are not the same size")
        print(f"MSE:\t{sum(self.MSE_l)/len(self.MSE_l)}\n"+f"Accuracy:\t{sum(self.accuracy_l)/len(self.accuracy_l)}")
        self.accuracy_l = []
        self.MSE_l = []
        
    def evaluate(self, X: list, y: list) -> None:
        """
        Test the network with a unseen test set of data

        Args:
            X (list): data set used for evaluating.
            y (list): targets or the data set.

        Raises:
            ValueError: Data set and Targets are not the same size
        """
        # quick check to see if the data and target list are the same size.
        if len(X) == len(y):

            for index in range(len(X)):

                # seprate the inputs and target values
                train_input = X[index]
                target = y[index]

                Y = self.predict(train_input)
                self.MSE_acc(Y, target)
            print(f"MSE:\t{sum(self.MSE_l)/len(self.MSE_l)}\n"+f"Accuracy:\t{sum(self.accuracy_l)/len(self.accuracy_l)}")
            self.accuracy_l = []
            self.MSE_l = []
        else:
            raise ValueError("Data set and Targets are not the same size")