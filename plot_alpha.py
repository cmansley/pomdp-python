import sys
import matplotlib.pyplot as plt
from collections import defaultdict

from itertools import imap
import operator
def dotproduct(vec1, vec2):
    """
    """
    return sum(imap(operator.mul, vec1, vec2))


if __name__ == "__main__":
    
    support = defaultdict(list)
    with open(sys.argv[1], 'r') as f:
        line = f.readline()
        while line:
            action = int(line)
            line = f.readline()
            support[action].append(map(float, line.split()))
            nl = f.readline()
            line = f.readline()


    fig, ax = plt.subplots(1)

    x = [ (1./20)*ii for ii in range(21) ]
    colors = ['blue', 'red', 'yellow', 'black']

    i = 0
    for key in support.keys():
        i += 1
        for v in support[key]: 
            y = map(lambda z: dotproduct([z, 1-z], v),x)
            ax.plot(x, y, color=colors[i])

fig.savefig('output.png')
