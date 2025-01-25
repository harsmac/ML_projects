import matplotlib.pyplot as plt
import numpy as np


def plot_lines(X, y):
    pass

def plot_moons(X, y, wts):
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")

    # X contains two features, x1 and x2
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)

    # Plotting the decision boundary
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = -(wts[0] + wts[1] * x1)
    plt.plot(x1, x2, "r--", label="Decision boundary")

    # Simplifying the plot by removing the axis scales.
    plt.xticks([])
    plt.yticks([])

    # Displaying the plot.
    # plt.savefig('img.png')
    plt.show()

def plot_circles(X, y, wts):
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")

    # X contains two features, x1 and x2
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)

    # Plotting the decision boundary
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = -(wts[0] + wts[1] * x1)
    plt.plot(x1, x2, "r--", label="Decision boundary")

    # Simplifying the plot by removing the axis scales.
    plt.xticks([])
    plt.yticks([])

    # Displaying the plot.
    plt.show()