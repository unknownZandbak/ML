from perceptron import Perceptron
import numpy as np

def train_perceptron(perceptron, TT, epochs)-> None:
    
    print(f"Perceptron Beginning values:\n {perceptron}")

    for epoch in range(epochs):
        print(f"\n=========Epoch {epoch+1}=========")

        for item in TT:
            # print(f"\n")
            
            i0 = int(item[0][0])
            i1 = int(item[0][1])

            Y = perceptron.activation([i0, i1])
            # print(f"")
            # print(f"Correcte Output: {item[1]}\n")
            print(f"Input: {i0, i1} || gekregen Output: {Y} || Correcte Output: {item[1]}")

            perceptron.update(item[1],Y,[i0, i1], .1)
            # print(f"volgende waarden na het trainen\n{perceptron}")
        total_loss = perceptron.loss(4)
        print(f"Total loss: {total_loss}")

        if total_loss == 0:
            print("\nTraining completed early\nStopping process")
            break
    print(f"\nuiteindelijke waarden:\n{perceptron}")

if __name__ == "__main__" :
    TT_and = np.array([
    [[0,0], 0],
    [[0,1], 0],
    [[1,0], 0],
    [[1,1], 1]])

    TT_xor = np.array([
    [[0,0], 0],
    [[0,1], 1],
    [[1,0], 1],
    [[1,1], 0]])

    AND = Perceptron([1,0], 1)
    XOR = Perceptron([4,-5], 1)
    
    train_perceptron(AND, TT_and, 20)
    train_perceptron(XOR, TT_xor, 20)