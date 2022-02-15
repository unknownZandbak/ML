import numpy as np
from perceptron import *


def test_AND_gate(inputM):
    AND = Perceptron((0.5, 0.5), -1)

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]

        AND.activation([i0, i1])
        return AND

def test_NAND_gate(inputM):
    NAND = Perceptron((-0.5, -0.5), 0.5)

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]

        NAND.activation([i0, i1])
        return NAND

def test_NOT_gate(inputM):
    NOT = Perceptron( (-1,), .5)

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]

        NOT.activation([i0,])
        return NOT

def test_OR_gate(inputM):
    OR = Perceptron( (0.5, 0.5), -.5)
    
    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]

        OR.activation([i0, i1])
        return OR

def test_NOR_gate(inputM):
    NOR = Perceptron((-1,-1,-1), 0)

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]
        i2 = i[2]

        NOR.activation([i0, i1, i2])
        return NOR

def test_Party_gate(inputM):
    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]
        i2 = i[2]

        PARTY = Perceptron((.6,.3,.2), -.4)
        PARTY.activation([i0, i1, i2])
        del PARTY

def test_XOR_gate(inputM):


    l1 = Perceptron_layer([NAND, OR])
    l2 = Perceptron_layer([AND,])

    XOR = Perceptro_network([l1,l2])

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]

        print(XOR.activation([i0,i1]))

def test_ADDR_gate(inputM):


    l1 = Perceptron_layer([NAND, OR, AND])
    l2 = Perceptron_layer([Perceptron([.5, .5, 0],-1), Perceptron([0, 0, 1], -.1)])

    ADDR = Perceptro_network([l1,l2])

    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = i[0]
        i1 = i[1]

        print(ADDR.activation([i0,i1]))

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

    # 
    # Network testing

    print("\nTesting XOR gate")
    XOR = test_XOR_gate(inputM2)


    print("\nTesting Half Adder gate")
    ADDR = test_ADDR_gate(inputM2)
