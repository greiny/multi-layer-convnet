import numpy as np
import matplotlib.pyplot as plt
import csv
import os.path

padding = 1
for num in range(10):
    for numimage in range(1000):
        imgname = 'kernel_%d/sample_%d.csv' % (num,numimage)
        if os.path.isfile(imgname):
            figname = 'kernel_%d/sample_%d' % (num,numimage)
            imgdata = np.genfromtxt(imgname, delimiter=",")

            # Assume the filters are squarnume
            filter_size = imgdata.shape[0]
            # Size of the result image including padding
            result_size = (filter_size + padding) - padding
            # Initialize result image to all zeros
            result = np.zeros((result_size, result_size))

            # Tile the filters into the result image
            filter_x = 0
            filter_y = 0

            for i in range(filter_size):
                for j in range(filter_size):
                    result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = imgdata[i, j]
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

