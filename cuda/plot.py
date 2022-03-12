import sys
import numpy as np
import matplotlib.pyplot as plt

def readFile(filename):
    f = open(filename)
    sizes = []
    times = []
    title = ''
    try:
        title = f.readline()
        # skip 3 line
        f.readline()
        f.readline()
        f.readline()
        while True:
            line = f.readline()
            if line:
                slices = line.split(" ")
                if len(slices) <= 2:
                    break;
                size = int(slices[0])
                time = float(slices[1])
                sizes.append(size)
                times.append(time)
    finally:
        f.close()
    return title, sizes, times

if __name__ == '__main__':
    plt.xlabel('shape')
    plt.ylabel('gflops')
    l = len(sys.argv)
    for i,item in enumerate(sys.argv):
        if i == 0:
            continue
        t,x,y = readFile(item)
        plt.plot(x,y,label=t)
    plt.legend()
    plt.show()

