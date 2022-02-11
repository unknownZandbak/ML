from perceptron import *

def test_AND_gate(inputM):
    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = Input(i[0])
        i1 = Input(i[1])

        P = Perceptron([i0, i1], (0.5, 0.5), 1)
        P.activation()

def test_NOT_gate(inputM):
    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = Input(i)

        P = Perceptron([i0,], (-1,), -.5)
        P.activation()

def test_OR_gate(inputM):
    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = Input(i[0])
        i1 = Input(i[1])

        P = Perceptron([i0, i1], (0.5, 0.5), .5)

        P.activation()

def test_NOR_gate(inputM):
    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = Input(i[0])
        i1 = Input(i[1])
        i2 = Input(i[2])

        P = Perceptron([i0, i1, i2], (-1,-1,-1), 0)

        P.activation()

def test_Party_gate(inputM):
    for i in inputM:
        print(f"\ninputs: {i}")
        i0 = Input(i[0])
        i1 = Input(i[1])
        i2 = Input(i[2])

        P = Perceptron([i0, i1, i2], (.6,.3,.2), .4)

        P.activation()

if __name__ == "__main__" :

    inputM1 = np.array([(0,),(1,)])
    inputM2 = np.array([(0,0),(0,1),(1,0),(1,1)])
    inputM3 = np.array([(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)])

    AND = Perceptron([(0,0),(0,1),(1,0),(1,1)], (.5, .5), 1)
    NOT = Perceptron([(0,)(1,)], (-1,), -.5)
    OR = Perceptron([(0,0),(0,1),(1,0),(1,1)], (0.5, 0.5), .5)

    print("\nTesting AND gate")
    test_AND_gate(inputM2)

    print("\nTesting Not gate")
    test_NOT_gate(inputM1)
    
    print("\nTesting OR gate")
    test_OR_gate(inputM2)

    print("\nTesting NOR gate")
    test_NOR_gate(inputM3)

    print("\nTesting Party gate")
    test_Party_gate(inputM3)