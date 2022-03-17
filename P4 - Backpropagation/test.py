from neuron import *

def test_fit_AND(epochs)-> None:
    """
        Test de backpropagation algoritme,
        door een netwerk te trainenen voor een AND logic gate 
    """
    print(f"\n#=#=#=#=#=#=#=# Training AND gate #=#=#=#=#=#=#=#")
    
    
    data = [[0,0],[0,1],[1,0],[1,1]]
    target = [[0], [0], [0], [1]]
    

    AND = Neuron_network(2, [1])
    AND.train(data, target, epochs)
    AND.evaluate(data, target)

    print(f"\n#=#=#=#=#=#=#=# Training Done #=#=#=#=#=#=#=#")

def test_fit_XOR(epochs)-> None:
    """
        Test de backpropagation algoritme,
        door een netwerk te trainenen voor een XOR logic gate 
    """
    print(f"\n#=#=#=#=#=#=#=# Training XOR gate #=#=#=#=#=#=#=#")
    
    data = [[0,0],[0,1],[1,0],[1,1]]
    target = [[0], [1], [1], [0]]

    XOR = Neuron_network(2, [2, 1])
    XOR.train(data, target, epochs)
    XOR.evaluate(data, target)

    print(f"\n#=#=#=#=#=#=#=# Training Done #=#=#=#=#=#=#=#")

def test_fit_ADDR(epochs)-> None:
    """
        Test de backpropagation algoritme,
        door een netwerk te trainenen voor een Half Adder logic gate 
    """
    print(f"\n#=#=#=#=#=#=#=# Training Half Adder #=#=#=#=#=#=#=#")

    data = [[0,0],[0,1],[1,0],[1,1]]
    target = [[0,0],[1,0],[1,0],[0,1]]

    ADDR = Neuron_network(2, [3, 2])
    ADDR.train(data, target, epochs)
    ADDR.evaluate(data, target)

    print(f"\n#=#=#=#=#=#=#=# Training Done #=#=#=#=#=#=#=#")

if __name__ == "__main__" :

    # test_fit_AND(2000)
    # test_fit_XOR(3000)
    test_fit_ADDR(5)