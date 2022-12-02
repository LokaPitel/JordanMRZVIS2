import numpy as np

if __name__ == '__main__':
    sequences = [
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
        [1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800],
        [0, 1, 0, -1, 0, 1, 0, -1, 0, 1],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ]

    k = 10 # Count of sequence elements
    e = 0.09 # Max error
    learning_rate = 0.0003
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

    W_first = np.zeros((train_col + 1, train_col * 2))
    W_second = np.zeros((train_col * 2, 1))

    W_first_delta = 0
    W_second_delta = 0

    error = 0

    for iteration in range(1, N + 1):
        error = 0

        for input_sequence in range(train_row):
            date = np.empty((1, train_col + 1))
            date[0][0:train_col] = train_matrix[input_sequence]
            date[0][-1] = Z
    
            hidden = date @ W_first
            out = hidden @ W_second
            Z = out.item()

            delta = out - train_etalons[input_sequence][0]

            W_first_delta = momentum * W_first_delta - learning_rate * delta * W_second.T
            W_second_delta = momentum * W_second_delta - learning_rate * delta

            W_first += W_first_delta
            W_second += W_second_delta
            
            date = np.empty((1, train_col + 1))
            date[0][0:train_col] = train_matrix[input_sequence]
            date[0][-1] = Z
    
            hidden = date @ W_first
            out = hidden @ W_second
            delta = out - train_etalons[input_sequence][0]

            error += (delta @ delta).item()
            #print(train_etalons[input_sequence][0])
            #print(out)

        print(f"Iteration #{iteration}: {error}")

