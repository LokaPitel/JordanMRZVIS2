from email import iterators
import numpy as np

class JordanNetwork:
    def __init__(self, p, L, e):
        self.e = e

        self.p = p
        self.L = L

        train_row, train_col = (p + 1), L

        self.W_first = np.random.uniform(size=(train_col + 1, train_col), low=-1, high=1)
        self.W_second = np.random.uniform(size=(train_col, 1), low=-1, high=1)

    def forward(self, sequence):
        assert len(sequence) == L

        data = np.empty((1, L + 1))
        data[0][0:L] = sequence
        data[0][-1] = Z

        hidden = data @ self.W_first
        out = hidden @ self.W_second

        return out

    def backpropogation(self, train_matrix, train_etalons):
        error = 10000000
        previous_error = 10000

        iteration_number = 0
        while abs(error - previous_error) > self.e:
            previous_error = error
            error = 0

            for input_sequence in range(len(train_matrix)):
                data = np.empty((1, L + 1))
                data[0][0:L] = train_matrix[input_sequence]
                data[0][-1] = Z

                hidden = data @ self.W_first
                out = hidden @ self.W_second

                delta = out - train_etalons[input_sequence][0]

                W_first_delta = -learning_rate * data.T @ delta @ self.W_second.T
                W_second_delta = -learning_rate * hidden.T @ delta

                self.W_first += W_first_delta
                self.W_second += W_second_delta

                
                data = np.empty((1, L + 1))
                data[0][0:L] = train_matrix[input_sequence]
                data[0][-1] = Z

                hidden = data @ self.W_first
                out = hidden @ self.W_second

                delta = out - train_etalons[input_sequence][0]
                error += (delta @ delta).item()

            iteration_number += 1
            print("Epoch #", iteration_number, "Loss: ", error)




def normalize_matrix(matrix):
    for i_f in range(len(matrix[0])):
        s = 0
        for j_f in range(len(matrix)):
            s += matrix[j_f][i_f] * matrix[j_f][i_f]
        s = np.sqrt(s)

        for j_f in range(len(matrix)):
            matrix[j_f][i_f] = matrix[j_f][i_f] / (s + 1)

    return matrix

if __name__ == '__main__':
    sequences = [
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
        [1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800],
        [0, 1, 0, -1, 0, 1, 0, -1, 0, 1],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ]

    k = 10 # Count of sequence elements
    e = 0.000009 # Max error
    learning_rate = 0.0007
    momentum = 0.1
    N = 1000000 # Count of epochs
    p = 4 # Window size
    L = (k - 1) - p

    Z = 0

    train_row, train_col = (p + 1), L

    train_matrix = np.empty((train_row, train_col))
    train_etalons = np.empty((train_row, 1))

    current_sequence = sequences[0]

    for i in range(train_row):
        for j in range(train_col):
            train_matrix[i, j] = current_sequence[i + j]

        train_etalons[i - 1][0] = train_matrix[i][train_col - 1]

    train_etalons[train_row - 1][0] = current_sequence[-1]

    print(train_etalons)
    
    print(train_matrix)

    W_first = np.random.uniform(size=(train_col + 1, train_col), low=-1, high=1)
    W_second = np.random.uniform(size=(train_col, 1), low=-1, high=1)

    W_first_delta = 0
    W_second_delta = 0

    error = 1000000
    iteration = 0

    #while error > e:
    #    error = 0

    #    iteration += 1

    #    for input_sequence in range(train_row):
    #        data = np.empty((1, train_col + 1))
    #        data[0][0:train_col] = train_matrix[input_sequence]
    #        data[0][-1] = Z
    
    #        hidden = data @ W_first
    #        out = hidden @ W_second
    #        Z = out.item()

    #        delta = out - train_etalons[input_sequence][0]

    #        W_first_delta = -learning_rate * data.T @ delta @ W_second.T
    #        W_second_delta = -learning_rate * hidden.T @ delta

    #        W_first += W_first_delta
    #        W_second += W_second_delta

    #        data = np.empty((1, train_col + 1))
    #        data[0][0:train_col] = train_matrix[input_sequence]
    #        data[0][-1] = Z
    
    #        hidden = data @ W_first
    #        out = hidden @ W_second
    #        delta = out - train_etalons[input_sequence][0]

    #        error += (delta @ delta).item()
    #        #print(train_etalons[input_sequence][0])
    #        #print(out)

    #    print(f"Iteration #{iteration}: {error}")

    network = JordanNetwork(4, k - 1 - p, 0)

    network.backpropogation(train_matrix, train_etalons)

    print(network.forward([0, 1, 1, 2, 3]))