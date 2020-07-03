import os
import time
import json
import argparse
import random

import matplotlib.pyplot as plt

from docrec.metrics import accuracy, Qc
from docrec.strips.strips import Strips
from docrec.compatibility.proposed import Proposed
from docrec.pipeline import Pipeline
from docrec.solver.solverconcorde import SolverConcorde
from docrec.solver.solverkbh import SolverKBH
try:
    from docrec.solver.solverlocal import SolverLS
except ModuleNotFoundError:
    print("No module named 'localsolver'. Please, check https://www.localsolver.com/docs/last/quickstart/solvingyourfirstmodelinpython.html for installing directions.")


NUM_CLASSES = 2

# parameters processing
parser = argparse.ArgumentParser(description='Reconstruction demo.')
parser.add_argument(
    '-d', '--doc', action='store', dest='doc', required=False, type=str,
    default='datasets/D2/mechanical/D010', help='Dataset [D1, D2].'
)
parser.add_argument(
    '-a', '--arch', action='store', dest='arch', required=False, type=str,
    default='squeezenet', help='Network architecture [squeezenet or mobilenet].'
)
parser.add_argument(
    '-s', '--solver', action='store', dest='solver', required=False, type=str,
    default='concorde', help='Optimizaiton solver [Concorde, KBH, LS].'
)
parser.add_argument(
    '-t', '--thresh', action='store', dest='thresh', required=False, type=str,
    default='otsu', help='Thresholding method [otsu or sauvola].'
)
parser.add_argument(
    '-se', '--seed', action='store', dest='seed', required=False, type=float,
    default=100, help='Seed (float) for the training process.'
)
args = parser.parse_args()

assert args.arch in ['squeezenet', 'mobilenet']
assert args.solver in ['concorde', 'kbh', 'LS']
assert args.thresh in ['otsu', 'sauvola']

random.seed(int(args.seed))

model_file_ext = 'npy' if args.arch == 'squeezenet' else 'h5'
best_epoch = json.load(open('traindata/{}/info.json'.format(args.arch), 'r'))['best_epoch']
weights_path = 'traindata/{}/model/{}.{}'.format(args.arch, best_epoch, model_file_ext)

# pipeline: compatibility algorithm + solver
width = 32 if args.arch == 'mobilenet' else 31
algorithm = Proposed(args.arch, weights_path, 10, (3000, width), num_classes=NUM_CLASSES, verbose=False, thresh=args.thresh)
if args.solver == 'concorde':
    solver = SolverConcorde(maximize=True, max_precision=2)
elif args.solver == 'kbh':
    solver = SolverKBH(maximize=True)
else:
    solver = SolverLS(maximize=True)
pipeline = Pipeline(algorithm, [solver])

# load strips and shuffle the strips
print('1) Load strips')
strips = Strips(path=args.doc, filter_blanks=True)
strips.shuffle()
init_permutation = strips.permutation()
print('Shuffled order: ' + str(init_permutation))

print('2) Results')
pipeline.run(strips)
# matrix -> list (displacements according the neighobors strips in solution)
compatibilities = pipeline.algorithm.compatibilities
displacements = pipeline.algorithm.displacements
solution = pipeline.solvers[0].solution
displacements = [displacements[prev][curr] for prev, curr in zip(solution[: -1], solution[1 :])]
corrected = [init_permutation[idx] for idx in solution]
print('Solution: ' + str(solution))
print('Correct order: ' + str(corrected))
print('Accuracy={:.2f}%'.format(100 * accuracy(solution, init_permutation)))
print('Qc={:.2f}%'.format(100 * Qc(compatibilities, init_permutation, pre_process=True, normalized=True)))
reconstruction = strips.image(order=solution, displacements=displacements)
plt.imshow(reconstruction, cmap='gray')
plt.axis('off')
plt.show()