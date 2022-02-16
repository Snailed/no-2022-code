import matplotlib.pyplot as plt
import numpy as np

class PlotController:
    @staticmethod
    def plot(f, label):
        x, y = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((100, 100))
        for i in range(0, 100):
            for j in range(0, 100):
                Z[i][j] = f([X[i][j], Y[i][j]])

        plt.axes(projection='3d')
        ax = plt.axes(projection="3d")
        ax.contour3D(X, Y, Z, 1000)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel(label)
        plt.show()
