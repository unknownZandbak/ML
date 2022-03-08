import numpy as np
from neuron import *

def test_fit_ADDR()-> Neuron:
    """
        Test de backpropagation algoritme,
        door een netwerk te trainenen voor een half adder logic gate 
    """
    print(f"\n#=#=#=#=#=#=#=# Training Half Adder #=#=#=#=#=#=#=#")



    print(f"\n#=#=#=#=#=#=#=# Training Done #=#=#=#=#=#=#=#")

if __name__ == "__main__" :

    ADDR = Neuron_network(2, [3, 2])
    print (ADDR.layers) 
    for layers in ADDR.layers:
        print(layers.neurons)