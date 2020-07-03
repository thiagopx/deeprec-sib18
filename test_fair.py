import sys
sys.path.append('libs') # Andalo requirements
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


# parameters processing
parser = argparse.ArgumentParser(description='Test Unfair')

parser.add_argument(
    '-se', '--seed', action='store', dest='seed', required=False, type=float,
    default=100, help='Seed (float) for the training process.'
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
solvers = [SolverConcorde(maximize=False, max_precision=2), SolverKBH(maximize=False)]

# reconstruction instances
docs1 = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
docs2 = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
docs = docs1 + docs2

# reconstruction instances
strips_all = {doc: Strips(path=doc, filter_blanks=True).shuffle() for doc in docs}

processed = 1
total = len(docs) * len(algorithms)
records = []
for algorithm in algorithms:
    d = 2 if algorithm.id() == 'marques' else 0
    for doc, strips in strips_all.items():
        print('[{:.2f}%] algorithm={} doc={}'.format(100 * processed / total, algorithm.id(), doc))
        processed += 1
        init_permutation = strips.permutation()
        compatibilities = algorithm(strips=strips, d=d).compatibilities
        qc = Qc(compatibilities, init_permutation, pre_process=False, normalized=True)
        for solver in solvers:
            solution = solver(instance=compatibilities).solution
            if solution is not None: # some solutions for the tested algorithm are None due to the poor compat. eval.
                acc = accuracy(solution, init_permutation)
                print('     => {} - acc={:.2f}% qc={:.2f}%'.format(solver.id(), 100 * acc, 100 * qc))
                records.append([algorithm.id(), solver.id(), doc, acc, qc, init_permutation, solution, compatibilities.tolist()])

os.makedirs('results', exist_ok=True)
json.dump(records, open('results/fair.json', 'w'))
print('Elapsed time={:.2f} sec.'.format(time.time() - t0_glob))