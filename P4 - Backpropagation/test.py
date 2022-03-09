import numpy as np
from neuron import *

def test_fit_AND()-> None:
    """
        Test de backpropagation algoritme,
        door een netwerk te trainenen voor een AND logic gate 
    """
    print(f"\n#=#=#=#=#=#=#=# Training AND gate #=#=#=#=#=#=#=#")
    
    TT = {
        "data"  : [[0,0],[0,1],[1,0],[1,1]],
        "target": [[0], [0], [0], [1]]
    }

    AND = Neuron_network(2, [1])
    AND.fit(TT, 20)

    print(f"\n#=#=#=#=#=#=#=# Training Done #=#=#=#=#=#=#=#")

def test_fit_XOR()-> None:
    """
        Test de backpropagation algoritme,
        door een netwerk te trainenen voor een XOR logic gate 
    """
    print(f"\n#=#=#=#=#=#=#=# Training XOR gate #=#=#=#=#=#=#=#")
    
    TT = {
        "data"  : [[0,0],[0,1],[1,0],[1,1]],
        "target": [[0], [1], [1], [0]]
    }

    XOR = Neuron_network(2, [2, 1])
    XOR.fit(TT, 20)

    print(f"\n#=#=#=#=#=#=#=# Training Done #=#=#=#=#=#=#=#")

def test_fit_ADDR()-> None:
    """
        Test de backpropagation algoritme,
        door een netwerk te trainenen voor een Half Adder logic gate 
    """
    print(f"\n#=#=#=#=#=#=#=# Training Half Adder #=#=#=#=#=#=#=#")
    
    TT = {
        "data"  : [[0,0],[0,1],[1,0],[1,1]],
        "target": [[0,0],[1,0],[1,0],[0,1]]
    }

    ADDR = Neuron_network(2, [3, 2])
    ADDR.fit(TT, 20)

    print(f"\n#=#=#=#=#=#=#=# Training Done #=#=#=#=#=#=#=#")

if __name__ == "__main__" :

    test_fit_AND()
    test_fit_XOR()
    test_fit_ADDR()