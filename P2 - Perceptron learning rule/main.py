from cupshelpers import Printer
from perceptron import Perceptron
import numpy as np

def test_learning_rule(TT, epochs)-> None:
    AND = Perceptron([1,0], 1)
    print(f"Perceptron Beginning values:\n {AND}")

    for epoch in range(epochs):
        print(f"\n=========Epoch {epoch+1}=========")

        for item in TT:
            # print(f"\n")
            
            i0 = int(item[0][0])
            i1 = int(item[0][1])

            Y = AND.activation([i0, i1])
            # print(f"")
            # print(f"Correcte Output: {item[1]}\n")
            print(f"Input: {i0, i1} || gekregen Output: {Y} || Correcte Output: {item[1]}")

            AND.update(item[1],Y,[i0, i1], .1)
            # print(f"volgende waarden na het trainen\n{AND}")
        total_loss = AND.loss(4)
        print(f"Total loss: {total_loss}")

        if total_loss == 0:
            print("\nTraining completed early\nStopping process")
            break

if __name__ == "__main__" :
    TT = np.array([
    [[0,0], 0],
    [[0,1], 0],
    [[1,0], 0],
    [[1,1], 1]])

    test_learning_rule(TT, 50)