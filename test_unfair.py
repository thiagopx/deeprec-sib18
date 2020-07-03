import sys
sys.path.append('docrec/libs') # Andalo requirements
import os
import time
import json
import argparse
import random
import matplotlib.pyplot as plt

from docrec.metrics import accuracy, Qc
from docrec.strips.strips import Strips

# compatibility algorithms
from docrec.compatibility.andalo import Andalo
from docrec.compatibility.marques import Marques
from docrec.compatibility.balme import Balme
from docrec.compatibility.morandell import Morandell
from docrec.compatibility.sleit import Sleit

# solvers
from docrec.solver.solverconcorde import SolverConcorde
from docrec.solver.solverkbh import SolverKBH
try:
    from docrec.solver.solverlocal import SolverLS
except ModuleNotFoundError:
    print("No module named 'localsolver'. Please, check https://www.localsolver.com/docs/last/quickstart/solvingyourfirstmodelinpython.html for installing directions.")


POOL_SIZE = 10

# parameters processing
parser = argparse.ArgumentParser(description='Test Unfair')

parser.add_argument(
    '-se', '--seed', action='store', dest='seed', required=False, type=float,
    default=0, help='Seed (float) for the training process.'
)
args = parser.parse_args()

t0_glob = time.time()
random.seed(int(args.seed))

# reconstruction pipeline (compatibility algorithm + solver)
algorithms = [Andalo(p=1.0, q=3.0),
              Marques(),
              Balme(tau=0.1),
              Morandell(epsilon=10, h=1, pi=0, phi=0),
              Sleit(t=0.15, h=0.25, p=0.33, linesth=1, blackth=2)]

# solver = [SolverLS(maximize=True)]
solver = SolverConcorde(maximize=False, max_precision=2)

# reconstruction instances
docs1 = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
docs2 = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
docs = docs1 + docs2

# reconstruction instances
strips_all = {doc: Strips(path=doc, filter_blanks=True).shuffle() for doc in docs}

processed = 1
total = len(docs) * len(algorithms) * POOL_SIZE
records = []
for algorithm in algorithms:
    for doc, strips in strips_all.items():
        print('[{:.2f}%] algorithm={} doc={} :: '.format(100 * processed / total, algorithm.id(), doc), end='')
        init_permutation = strips.permutation()
        solutions = []
        for d in range(0, POOL_SIZE):
            processed += 1
            compatibilities = algorithm(strips=strips, d=d).compatibilities
            solution = solver(instance=compatibilities).solution
            if solution is not None:
                solutions.append((solution, d, compatibilities, accuracy(solution, init_permutation)))

        # keep only the best solution for the algorithm/document
        best_solution, best_d, best_compatibilities, max_accuracy = max(solutions, key=lambda item: item[3])
        qc = Qc(best_compatibilities, init_permutation, pre_process=False, normalized=True)
        print('acc={:.2f}% qc={:.2f}% best_d={}'.format(100 * max_accuracy, 100 * qc, best_d))
        records.append([algorithm.id(), solver.id(), doc, max_accuracy, qc, init_permutation, solution, best_compatibilities.tolist(), best_d])

os.makedirs('results', exist_ok=True)
json.dump(records, open('results/unfair.json', 'w'))
print('Elapsed time={:.2f} sec.'.format(time.time() - t0_glob))
