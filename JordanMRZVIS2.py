import random
import numpy as np
import math

def factorial(n):
    if n < 0:
        return None

    elif n == 0 or n == 1:
        return 1

    return n * factorial(n - 1)

def fib(n):
    if n == 0:
        return 0

    elif n == 1:
        return 1

    return fib(n - 1) + fib(n - 2)

def sin(n):
    return round(math.sin(n * math.pi / 2))

def pow_fun(n):
    return 1.125 ** n

def fact_fun(n):
    return math.log(factorial(n))


class JordanNetwork:
    def __init__(self, p, iterations_count, learning_rate):
        self.iterations_count = iterations_count

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        self.ilayer_size = p + 1
        self.hlayer_size = int(self.ilayer_size / 2)
        self.olayer_size = self.ilayer_size
         
        self.w_size = p + 1

        self.m_tb_first = 0
        self.v_tb_first = 0

        self.m_tb_second = 0
        self.v_tb_second = 0

        self.m_t_first = 0
        self.v_t_first = 0

        self.m_t_second = 0
        self.v_t_second = 0

        self.m_t_third = 0
        self.v_t_third = 0

        self.prevDb1 = 0
        self.prevDb2 = 0

        self.prevDW1 = 0
        self.prevDW2 = 0
        self.prevDW3 = 0

        self.GW1 = 0
        self.GW2 = 0
        self.GW3 = 0

        self.learning_rate = learning_rate

        self.previous_output = np.empty((1, self.olayer_size))

        self.b_first = np.random.uniform(size=(1, self.hlayer_size), low=-1, high=1)
        self.bW_first = np.ones((self.hlayer_size, self.hlayer_size))

        self.b_second = np.random.uniform(size=(1, self.olayer_size), low=-1, high=1)
        self.bW_second = np.ones((self.olayer_size, self.olayer_size))

        self.W_first = np.random.uniform(size=(self.ilayer_size, self.hlayer_size), low=-1, high=1)
        self.W_second = np.random.uniform(size=(self.hlayer_size, self.olayer_size), low=-1, high=1)
        self.W_third = np.random.uniform(size=(self.ilayer_size, self.hlayer_size), low=-1, high=1)

    def forward(self, sequence):
        data = np.empty((1, self.ilayer_size))
        data[0][...] = sequence

        hidden = data @ self.W_first + self.previous_output @ self.W_third + self.b_first @ self.bW_first
        out = hidden @ self.W_second + self.b_second @ self.bW_second

        return out

    def backpropogation(self, train_matrix, train_etalons):
        error = 10000000
        previous_error = 10000

        count_of_coinc = 0

        iteration_number = 0
        while iteration_number < self.iterations_count:
            previous_error = error
            error = 0

            for input_sequence in range(len(train_matrix)):
                data = np.empty((1, self.ilayer_size))
                data[0][...] = train_matrix[input_sequence]

                hidden = data @ self.W_first + self.previous_output @ self.W_third + self.b_first @ self.bW_first
                out = hidden @ self.W_second + self.b_second @ self.bW_second

                delta = out - train_etalons[input_sequence]

                b_first_delta = delta @ self.W_second.T
                b_second_delta = delta

                W_first_delta =  data.T @ delta @ self.W_second.T
                W_second_delta = hidden.T @ delta
                W_third_delta = self.previous_output.T @ delta @ self.W_second.T

                self.previous_output = out


                self.m_tb_first = self.beta1 * self.m_tb_first + (1 - self.beta1) * b_first_delta
                self.v_tb_first = self.beta2 * self.v_tb_first + (1 - self.beta2) * b_first_delta ** 2

                self.m_tb_second = self.beta1 * self.m_tb_second + (1 - self.beta1) * b_second_delta
                self.v_tb_second = self.beta2 * self.v_tb_second + (1 - self.beta2) * b_second_delta ** 2

                self.m_t_first = self.beta1 * self.m_t_first + (1 - self.beta1) * W_first_delta
                self.v_t_first = self.beta2 * self.v_t_first + (1 - self.beta2) * W_first_delta ** 2

                self.m_t_second = self.beta1 * self.m_t_second + (1 - self.beta1) * W_second_delta
                self.v_t_second = self.beta2 * self.v_t_second + (1 - self.beta2) * W_second_delta ** 2

                self.m_t_third = self.beta1 * self.m_t_third + (1 - self.beta1) * W_third_delta
                self.v_t_third = self.beta2 * self.v_t_third + (1 - self.beta2) * W_third_delta ** 2

                self.prevDb1 = - learning_rate / (np.sqrt(self.v_tb_first) + self.eps) * self.m_tb_first
                self.prevDb2 = - learning_rate / (np.sqrt(self.v_tb_second) + self.eps) * self.m_tb_second

                self.prevDW1 = - learning_rate / (np.sqrt(self.v_t_first) + self.eps) * self.m_t_first
                self.prevDW2 = - learning_rate / (np.sqrt(self.v_t_second) + self.eps) * self.m_t_second
                self.prevDW3 = - learning_rate / (np.sqrt(self.v_t_third) + self.eps) * self.m_t_third

                self.b_first += self.prevDb1
                self.b_second += self.prevDb2

                self.W_first += self.prevDW1
                self.W_second += self.prevDW2
                self.W_third += self.prevDW3    


                data[0][...] = train_matrix[input_sequence]

                hidden = data @ self.W_first + self.previous_output @ self.W_third + self.b_first @ self.bW_first
                out = hidden @ self.W_second + self.b_second @ self.bW_second

                delta = out - train_etalons[input_sequence]
                error += delta @ delta.T

            iteration_number += 1
            print("Epoch #", iteration_number, "Loss: ", error)
               

if __name__ == '__main__':
    k = 10

    sequences = [
        [fib(i) for i in range(k)],
        [fact_fun(i) for i in range(k)],
        [sin(i) for i in range(k)],
        [pow_fun(i) for i in range(k)]
    ]

    sequence_functions = [fib, fact_fun, sin, pow_fun]

    sequence_names = ['Fibonacci', 'log_2 n!', 'Sin(i * pi / 2)', '1.125^n']

    q = k - 1
    p = 1
    L = q - p 

    learning_rate = 0.0007

    N = 200000 

    print("Sequences:")
    for index, i in enumerate(sequences):
        print(f"{index + 1})", i)

    choice = int(input("Select sequence: "))

    if (choice < 1):
        choice = 1

    elif (choice > len(sequences)):
        choice = len(sequences)
    
    current_sequence = sequences[choice - 1]
    
    train_row, train_col = L, p + 1 

    train_matrix = np.empty((train_row, train_col))
    train_etalons = np.empty((train_row, train_col))

    for i in range(train_row):
        for j in range(train_col):
            train_matrix[i, j] = current_sequence[i + j]
            train_etalons[i, j] = current_sequence[i + j + 1]

    network = JordanNetwork(p, N, learning_rate)
    network.backpropogation(train_matrix, train_etalons)

    while True:
        start = random.randint(k, k + 20)

        sequence = [sequence_functions[choice - 1](i) for i in range(start, start + p + 1)]

        print("Sequence name -", sequence_names[choice - 1])
        print("Sequence for test", sequence)

        print("Network output: ", network.forward(sequence))
        print("What you should see: ", [sequence_functions[choice - 1](i) for i in range(start + 1, start + p + 2)])

        answer = input("Do you want to repeat test?[Y/N]")

        if (answer == 'N' or answer == 'n'):
            break