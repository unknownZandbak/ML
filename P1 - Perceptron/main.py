from ast import Del
import numpy as np
from perceptron import *


def test_AND_gate(inputM):
    AND = Perceptron((0.5, 0.5), -1)

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]

        output = AND.activation([i0, i1])
        print(f"Output: {output}")

        # return de perceptron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return AND

def test_NAND_gate(inputM):
    NAND = Perceptron((-0.5, -0.5), 0.5)

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]

        output = NAND.activation([i0, i1])
        print(f"Output: {output}")
        
        # return de perceptron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return NAND

def test_NOT_gate(inputM):
    NOT = Perceptron( (-1,), .5)

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]

        output = NOT.activation([i0,])
        print(f"Output: {output}")
        
        # return de perceptron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return NOT

def test_OR_gate(inputM):
    OR = Perceptron( (0.5, 0.5), -.5)
    
    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]

        output = OR.activation([i0, i1])
        print(f"Output: {output}")
        
        # return de perceptron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return OR

def test_NOR_gate(inputM):
    NOR = Perceptron((-1,-1,-1), 0)

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]
        i2 = i[2]

        output = NOR.activation([i0, i1, i2])
        print(f"Output: {output}")
        
        # return de perceptron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return NOR

def test_Party_gate(inputM):
    PARTY = Perceptron((.6,.3,.2), -.4)

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]
        i2 = i[2]

        output = PARTY.activation([i0, i1, i2])
        print(f"Output: {output}")


def test_XOR_gate(inputM):
    # Maak de lagen voor het netwerk
    l1 = Perceptron_layer([NAND, OR])
    l2 = Perceptron_layer([AND,])

    # Dan voegen we die lagen to aan het netwerk
    XOR = Perceptro_network([l1,l2])

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]

        output = XOR.activation([i0,i1])
        print(f"Output: {output}")
        
        # return de perceptron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return XOR
        
def test_ADDR_gate(inputM):
    # Maak de lagen voor het netwerk
    l1 = Perceptron_layer([NAND, OR, AND])
    l2 = Perceptron_layer([Perceptron([.5, .5, 0],-1), Perceptron([0, 0, 1], -.1)])

    # Dan voegen we die lagen to aan het netwerk
    ADDR = Perceptro_network([l1,l2])

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]
        # output van de Half Adder ziet er uit als een tuple(0(sum), 0(carry))
        output = ADDR.activation([i0,i1])
        print(f"Output: {output}")

        # return de perceptron zodat we die later kunnen simpel kunnen gebruiken in een netwerk.
    return ADDR

if __name__ == "__main__" :

    inputM1 = np.array([(0,),(1,)])
    inputM2 = np.array([(0,0),(0,1),(1,0),(1,1)])
    inputM3 = np.array([(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)])

    print("\nTesting AND gate")
    AND = test_AND_gate(inputM2)

    print("\nTesting NAND gate")
    NAND = test_NAND_gate(inputM2)

    print("\nTesting Not gate")
    NOT = test_NOT_gate(inputM1)
    
    print("\nTesting OR gate")
    OR = test_OR_gate(inputM2)

    print("\nTesting NOR gate")
    NOR = test_NOR_gate(inputM3)

    print("\nTesting Party gate")
    test_Party_gate(inputM3)

    # # 
    # # Network testing

    print("\nTesting XOR gate")
    XOR = test_XOR_gate(inputM2)


    print("\nTesting Half Adder gate")
    ADDR = test_ADDR_gate(inputM2)
