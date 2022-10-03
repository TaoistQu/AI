import numpy as np
import matplotlib.pylab as plt

def step_function(X):
    return np.array(X > 0 ,dtype=np.int32)


def sigmoid(X) :
    return 1/(1+np.exp(-X))

if __name__ == "__main__":
    X = np.arange(-5.0,5.0,0.1)
    Y1 = step_function(X)
    Y2 = sigmoid(X)
    plt.plot(X,Y1)
    plt.plot(X,Y2,'k--')

    plt.ylim(-0.1,1.1)
    plt.savefig("sig_step.png")
    plt.show()