import numpy as np
import matplotlib.pyplot as plt



def read_file(path):
    file = open(path, 'r')

    f = file.readlines()
    n = int(f[0].strip())

    indexes = []
    freqs = []

    for i in range(1, n+1):
        index = int(f[i].strip().split()[0])
        freq = float(f[i].strip().split()[1])

        indexes.append(index)
        freqs.append(freq)

    return indexes, freqs

def vis(x, y):
    plt.figure('fig1')
    plt.plot(x, y, marker = 'o')
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()





indexes, freqs = read_file("signal1.txt")
vis(indexes, freqs)