import numpy as np
from neuron import *


def test_AND_gate(inputM)-> Neuron:
    """
        Test een AND gate neuron
    """
    AND = Neuron((0.5, 0.5), -1)

    # Kleine truth table
    tt = [
        0,
        0,
        0,
        1
    ]

    for i in range(len(inputM)):
        i0 = inputM[i][0]
        i1 = inputM[i][1]

        output = AND.activation([i0, i1])
        print(f"Inputs: {inputM[i]} \t Gekregen output: {output} \t Verwachte Output {tt[i]}")

        # return de Neuron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return AND

def test_NAND_gate(inputM)-> Neuron:
    """
        Test een NAND gate neuron
    """
    NAND = Neuron((-0.5, -0.5), 0.5)
    
    # Kleine truth table
    tt = [
        1,
        1,
        1,
        0
    ]

    for i in range(len(inputM)):
        i0 = inputM[i][0]
        i1 = inputM[i][1]

        output = NAND.activation([i0, i1])
        print(f"Inputs: {inputM[i]} \t Gekregen output: {output} \t Verwachte Output {tt[i]}")
        
        # return de Neuron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return NAND

def test_NOT_gate(inputM)-> Neuron:
    """
        Test een NOT gate neuron
    """
    NOT = Neuron( (-1,), .5)

    # Kleine truth table
    tt = [
        1,
        0,
        ]

    for i in range(len(inputM)):
        i0 = inputM[i][0]

        output = NOT.activation([i0,])
        print(f"Inputs: {inputM[i]} \t Gekregen output: {output} \t Verwachte Output {tt[i]}")
        
        # return de Neuron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return NOT

def test_OR_gate(inputM)-> Neuron:
    """
        Test een OR gate neuron
    """
    OR = Neuron( (0.5, 0.5), -.5)

    # Kleine truth table
    tt = [
        0,
        1,
        1,
        1
    ]
    
    for i in range(len(inputM)):
        i0 = inputM[i][0]
        i1 = inputM[i][1]

        output = OR.activation([i0, i1])
        print(f"Inputs: {inputM[i]} \t Gekregen output: {output} \t Verwachte Output {tt[i]}")
        
        # return de Neuron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return OR

def test_NOR_gate(inputM)-> Neuron:
    """
        Test een NOR gate neuron
    """
    NOR = Neuron((-1,-1,-1), 0)

    # Kleine truth table
    tt = [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]

    for i in range(len(inputM)):
        i0 = inputM[i][0]
        i1 = inputM[i][1]
        i2 = inputM[i][2]

        output = NOR.activation([i0, i1, i2])
        print(f"Inputs: {inputM[i]} \t Gekregen output: {output} \t Verwachte Output {tt[i]}")
        
        # return de Neuron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return NOR
        
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

    inputM1 = np.array([(0,),(1,)])
    inputM2 = np.array([(0,0),(0,1),(1,0),(1,1)])
    inputM3 = np.array([(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)])

    print("\n=========Testing AND gate=========")
    AND = test_AND_gate(inputM2)

    print("\n=========Testing NAND gate=========")
    NAND = test_NAND_gate(inputM2)

    print("\n=========Testing Not gate=========")
    NOT = test_NOT_gate(inputM1)
    
    print("\n=========Testing OR gate=========")
    OR = test_OR_gate(inputM2)

    print("\n=========Testing NOR gate=========")
    NOR = test_NOR_gate(inputM3)

    # 
    # Network testing

    print("\n=========Testing Half Adder gate=========")
    ADDR = test_ADDR_gate(inputM2)
