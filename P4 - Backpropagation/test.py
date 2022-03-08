import numpy as np
from neuron import *

def test_ADDR_gate(inputM)-> Neuron:
    """
        Test een Half adder gate neuraal netwerk
    """
    # Maak de lagen voor het netwerk
    l1 = Neuron_layer([NAND, OR, AND])
    l2 = Neuron_layer([Neuron([.5, .5, 0],-1), Neuron([0, 0, 1], -.1)])

    # Dan voegen we die lagen to aan het netwerk
    ADDR = Neuron_network([l1,l2])

    # Kleine truth table
    tt = [
        [0,0],
        [1,0],
        [1,0],
        [0,1]
    ]

    for i in range(len(inputM)):
        i0 = inputM[i][0]
        i1 = inputM[i][1]

        # output van de Half Adder ziet er uit als een tuple(0(sum), 0(carry))
        output = ADDR.feed_forward([i0,i1])
        print(f"Inputs: {inputM[i]} \t Gekregen output: {output} \t Verwachte Output {tt[i]}")

        # return de Neuron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return ADDR


if __name__ == "__main__" :

    OR = Neuron( (0.5, 0.5), -.5)
    NAND = Neuron((-0.5, -0.5), 0.5)
    AND = Neuron((0.5, 0.5), -1)

    inputM1 = np.array([(0,),(1,)])
    inputM2 = np.array([(0,0),(0,1),(1,0),(1,1)])
    inputM3 = np.array([(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)])

    print("\n=========Testing Half Adder gate=========")
    ADDR = test_ADDR_gate(inputM2)
