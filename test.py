import sys
sys.path.append('libs') # Andalo requirements
import os
import time
import json
import argparse
import random
import matplotlib.pyplot as plt

from docrec.metrics import accuracy
from docrec.strips.strips import Strips
from docrec.pipeline import Pipeline

# compatibility algorithms
from docrec.compatibility.proposed import Proposed
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
parser = argparse.ArgumentParser(description='Test')

parser.add_argument(
    '-se', '--seed', action='store', dest='seed', required=False, type=float,
    default=100, help='Seed (float) for the training process.'
)
args = parser.parse_args()

t0_glob = time.time()
random.seed(int(args.seed))

# reconstruction pipeline (compatibility algorithm + solver)
best_epoch_sn = json.load(open('traindata/squeezenet/info.json', 'r'))['best_epoch']
best_epoch_mn = json.load(open('traindata/mobilenet/info.json', 'r'))['best_epoch']
weights_path_sn = 'traindata/squeezenet/model/{}.npy'.format(best_epoch_sn)
weights_path_mn = 'traindata/mobilenet/model/{}.h5'.format(best_epoch_mn)
proposed_sn = Proposed('squeezenet', weights_path_sn, 10, (3000, 31), num_classes=2, verbose=False, thresh='otsu')
proposed_mn = Proposed('mobilenet', weights_path_mn, 10, (3000, 32), num_classes=2, verbose=False, thresh='otsu')
andalo = Andalo(p=1.0, q=3.0)
marques = Marques()
balme = Balme(tau=0.1)
morandell = Morandell(epsilon=10, h=1, pi=0, phi=0)
sleit = Sleit(t=0.15, h=0.25, p=0.33, linesth=1, blackth=2)

# solvers_max = [SolverLS(maximize=True), SolverKBH(maximize=True)]
# solvers_min = [SolverLS(maximize=False), SolverKBH(maximize=False)]
solvers_max = [SolverConcorde(maximize=True, max_precision=2), SolverKBH(maximize=True)]
solvers_min = [SolverConcorde(maximize=False, max_precision=2), SolverKBH(maximize=False)]
pipelines = [Pipeline(proposed_sn, solvers_max),
             Pipeline(proposed_mn, solvers_max),
             Pipeline(andalo, solvers_min),
             Pipeline(marques, solvers_min),
             Pipeline(balme, solvers_min),
             Pipeline(morandell, solvers_min),
             Pipeline(sleit, solvers_min)]

# reconstruction instances
docs1 = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
docs2 = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
docs = docs1 + docs2

processed = 1
total = len(docs) * len(pipelines)
results = dict()
for doc in docs:
    t0 = time.time()
    strips = Strips(path=doc, filter_blanks=True)
    strips.shuffle()
    init_permutation = strips.permutation()
    t_load = time.time() - t0
    results[doc] = dict(init_permutation=init_permutation, time=t_load, algorithms=dict())
    for pipeline in pipelines:
        print('[{:.2f}%] algorithm={} doc={} ::'.format(100 * processed / total, pipeline.algorithm.id(), doc), end='')
        processed += 1
        d = 2 if pipeline.algorithm.id() == 'marques' else 0
        pipeline.run(strips, d)
        results[doc]['algorithms'][pipeline.algorithm.id()] = dict()
        results[doc]['algorithms'][pipeline.algorithm.id()]['time'] = pipeline.t_algorithm
        results[doc]['algorithms'][pipeline.algorithm.id()]['compatibilities'] = pipeline.algorithm.compatibilities.tolist()
        results[doc]['algorithms'][pipeline.algorithm.id()]['solvers'] = dict()

        acc_str = ''
        for solver, t_solver in zip(pipeline.solvers, pipeline.t_solvers):
            results[doc]['algorithms'][pipeline.algorithm.id()]['solvers'][solver.id()] = dict()
            results[doc]['algorithms'][pipeline.algorithm.id()]['solvers'][solver.id()]['solution'] = solver.solution
            results[doc]['algorithms'][pipeline.algorithm.id()]['solvers'][solver.id()]['time'] = t_solver
            acc_str += ' {}={:.2f}%'.format(solver.id(), 100 * accuracy(solver.solution, init_permutation))
        print(acc_str)

os.makedirs('results', exist_ok=True)
json.dump(results, open('results/default.json', 'w'))
print('Elapsed time={:.2f} sec.'.format(time.time() - t0_glob))
