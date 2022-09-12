import matplotlib.pyplot as plt
import numpy as np


class Field:
    def __init__(self, *args):
        if isinstance(args[0], tuple):
            self.shape = args[0]
            self.elements = np.zeros(args[0])
        elif isinstance(args[0], (list, np.ndarray)):
            self.elements = np.array(args[0])
            self.shape = self.elements.shape
        else:
            raise Exception(f"Input Type Error: {type(args[0])}")

        self.dim = np.ndim(self.elements)

    def __add__(self, other):
        if isinstance(other, Field):
            if other.shape == self.shape:
                result = self.copy()
                result.elements += other.elements
                return result
        elif isinstance(other, (int, float)):
            result = self.copy()
            result.elements += other
            return result

        raise Exception(f"Can not add types {type(self).__name__} and {type(other).__name__}")

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = self.copy()
            result.elements *= other
            return result

    def __rmul__(self, other):
        return self * other

    def copy(self):
        return Field(self.elements.copy())


class ScalarField(Field):
    def copy(self):
        return ScalarField(self.elements.copy())

    def display(self, cmap="inferno"):
        plt.imshow(self.elements, cmap=cmap)
        plt.show()

    def grad(self):
        gradient = np.stack([np.gradient(self.elements, axis=i) for i in range(self.dim)][::-1], axis=self.dim)
        return VectorField(gradient)

    def laplacian(self):
        return self.grad().div()


class VectorField(Field):
    def copy(self):
        return VectorField(self.elements.copy())

    def div(self):
        gradients = [np.gradient(self.elements[..., i], axis=self.shape[-1] - i - 1) for i in range(self.shape[-1])]
        divergence = np.sum(gradients, axis=0)
        return ScalarField(divergence)
