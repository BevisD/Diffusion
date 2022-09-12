from vectorcalc import ScalarField, VectorField
import numpy as np
import matplotlib.pyplot as plt

class Diffusion:
    def __init__(self, elements, diffusivity, dt=0.01):
        self.phi = ScalarField(elements)
        self.diffusivity = diffusivity
        self.dt = dt

    def display(self):
        self.phi.display()

    def update(self):
        grad = self.phi.grad()

        # NEUMANN BOUNDARY CONDITIONS
        # grad.elements[0] = 0  # TOP
        grad.elements[-1] = 0  # BOTTOM
        # grad.elements[:, 0] = 0  # LEFT
        # grad.elements[:, -1] = 0  # RIGHT
        deltas = grad.div()
        self.phi += deltas * self.dt * self.diffusivity

        #DIRICHLET BOUNDARY CONDITIONS
        self.phi.elements[0] = 0  # TOP
        # self.phi.elements[-1] = 0  # BOTTOM
        self.phi.elements[:, 0] = 1  # LEFT
        self.phi.elements[:, -1] = 1  # RIGHT

if __name__ == "__main__":
    ITERATIONS_PER_FRAME = 1000
    ANIMATION_FRAMES = 100

    COLS = 128
    ROWS = 128

    initial_conditions = np.array([[1 if (abs(i-COLS/2) < COLS/4 and abs(j-ROWS/2) < ROWS/4) else 0 for i in range(COLS)] for j in range(ROWS)], dtype=np.float64)
    diffusion_space = Diffusion(initial_conditions, 1, dt=0.01)

    im = plt.imshow(diffusion_space.phi.elements, cmap="inferno")

    totals = []
    for i in range(ANIMATION_FRAMES):
        im.set_data(diffusion_space.phi.elements)
        plt.title(f"Frame: {i}")
        plt.pause(0.001)
        for j in range(ITERATIONS_PER_FRAME):
            totals.append(np.sum(diffusion_space.phi.elements))
            diffusion_space.update()

    plt.pause(2)
    plt.clf()

    plt.title("Plot of total heat per iteration")
    plt.ylabel("Fractional change in heat")
    plt.xlabel("Iteration Number")
    plt.plot(totals/totals[0]-1)
    plt.show()
