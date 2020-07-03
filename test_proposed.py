import os
import time
import json
import argparse
import random
import matplotlib.pyplot as plt

from docrec.metrics import accuracy, Qc
from docrec.strips.strips import Strips

# compatibility algorithms
from docrec.compatibility.proposed import Proposed

# solvers
from docrec.solver.solverconcorde import SolverConcorde
from docrec.solver.solverkbh import SolverKBH
try:
    from docrec.solver.solverlocal import SolverLS
except ModuleNotFoundError:
    print("No module named 'localsolver'. Please, check https://www.localsolver.com/docs/last/quickstart/solvingyourfirstmodelinpython.html for installing directions.")


# parameters processing
parser = argparse.ArgumentParser(description='Test Proposed')

parser.add_argument(
    '-se', '--seed', action='store', dest='seed', required=False, type=float,
    default=0, help='Seed (float) for the training process.'
)
parser.add_argument(
    '-a', '--arch', action='store', dest='arch', required=False, type=str,
    default='squeezenet', help='Network architecture [squeezenet or mobilenet].'
)
args = parser.parse_args()

assert args.arch in ['squeezenet', 'mobilenet']

t0_glob = time.time()
random.seed(int(args.seed))

# reconstruction pipeline (compatibility algorithm + solver)
best_epoch = json.load(open('traindata/{}/info.json'.format(args.arch), 'r'))['best_epoch']
weights_path = 'traindata/{}/model/{}.{}'.format(args.arch, best_epoch, 'npy' if args.arch == 'squeezenet' else 'h5')
width = 31 if args.arch == 'squeezenet' else 32
algorithm = Proposed(args.arch, weights_path, 10, (3000, width), num_classes=2, verbose=False, thresh='otsu')

# solver = [SolverLS(maximize=True)]
solvers = [SolverConcorde(maximize=True, max_precision=2), SolverKBH(maximize=True)]

# reconstruction instances
docs1 = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
docs2 = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
docs = docs1 + docs2

# reconstruction instances
strips_all = {doc: Strips(path=doc, filter_blanks=True).shuffle() for doc in docs}

processed = 1
total = len(docs)
records = []
for doc, strips in strips_all.items():
    print('[{:.2f}%] algorithm={} doc={} :: '.format(100 * processed / total, algorithm.id(), doc), end='')
    processed += 1
    init_permutation = strips.permutation()
    compatibilities = algorithm(strips=strips).compatibilities
    qc = Qc(compatibilities, init_permutation, pre_process=True, normalized=True)
    for solver in solvers:
        solution = solver(instance=compatibilities).solution
        acc = accuracy(solution, init_permutation)
        print('     => {} - acc={:.2f}% qc={:.2f}%'.format(solver.id(), 100 * acc, 100 * qc))
        records.append([algorithm.id(), solver.id(), doc, acc, qc, init_permutation, solution, compatibilities.tolist()])

os.makedirs('results', exist_ok=True)
json.dump(records, open('results/proposed_{}.json'.format(args.arch), 'w'))
print('Elapsed time={:.2f} sec.'.format(time.time() - t0_glob))