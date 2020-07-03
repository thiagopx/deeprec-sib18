import numpy as np
from time import time


class Pipeline:

    def __init__(self, algorithm, solvers=[]):

        assert len(solvers) > 0

        self.algorithm = algorithm
        self.solvers = solvers
        self.t_algorithm = 0
        self.t_solvers = len(solvers) * [0]


    def run(self, strips, d=0):

        t0 = time()
        self.algorithm(strips=strips, d=d)
        self.t_algorithm = time() - t0

        for i, solver in enumerate(self.solvers):
            t0 = time()
            solver(instance=self.algorithm.compatibilities)
            self.t_solvers[i] = time() - t0
