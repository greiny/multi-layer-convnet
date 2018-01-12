import numpy as np
import matplotlib.pyplot as plt
import csv
import os.path

padding = 1

for num in range(10):
    filename = 'kernel_%d/ch_0.csv' % num
    if os.path.isfile(filename) :
        figname = 'kernel_%d/kernel_%d' % (num,num)
        data = np.genfromtxt(filename, delimiter=",")

        # Assume he filters are squarnume
        filter_size = data.shape[1]
        # Size of the result image including padding
        result_size = (filter_size + padding) - padding
        # Initialize result image to all zeros
        result = np.zeros((result_size, result_size))

        # Tile the filters into the result image
        filter_x = 0
        filter_y = 0

        for i in range(filter_size):
            for j in range(filter_size):
                result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = data[i, j]
        filter_x += 1

        # Normalize image to 0-1
        min = result.min()
        max = result.max()
        result = (result - min) / (max - min)

        # Plot figure
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(result, cmap='gray', interpolation='nearest')

        # Save plot if filename is set
        if figname != '':
            plt.savefig(figname, bbox_inches='tight', pad_inches=0)

