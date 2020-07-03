import sys
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
import pandas as pd

import seaborn as sns
sns.set(context='paper', style='whitegrid', palette='deep', font_scale=1.5)


# algorithms = ['proposed_squeezenet', 'proposed_mobilenet', 'andalo', 'morandell', 'balme', 'sleit', 'marques']
proposed_squeezenet = json.load(open('results/proposed_squeezenet.json', 'r'))
proposed_mobilenet = json.load(open('results/proposed_mobilenet.json', 'r'))

records = []
for entry in proposed_squeezenet + proposed_mobilenet:
    algorithm, solver, doc, acc = entry[: 4]
    algorithm_short = 'SN' if algorithm == 'proposed_squeezenet' else 'MN'
    dataset = doc.split('/')[1]
    records.append([algorithm_short + '/' + dataset, solver, 100 * acc])

df = pd.DataFrame.from_records(records, columns=('arch/dataset', 'solver', 'accuracy'))
meanlineprops = dict(linestyle='--', linewidth=1, color='red')
fp = sns.catplot(
    x='arch/dataset', y='accuracy', data=df,
    hue='solver', kind='box', hue_order=['KBH','Concorde'], height=3, aspect=2,
    margin_titles=True, fliersize=1.0, width=0.6, linewidth=1,
    legend=True, showmeans=True, meanline=True, meanprops=meanlineprops
)
# print(df['Accuracy (\\%)'].min())
yticks = [40, 50, 60, 70, 80, 90, 100]
fp.set(yticks=yticks, ylim=(min(yticks) - .5, max(yticks) + .5))
fp.set_yticklabels(yticks, fontdict={'fontsize': 15})
fp.ax.set_xlabel('Network/Dataset', fontsize=15)
fp.ax.set_ylabel('Accuracy (\%)', fontsize=15)
fp.despine(left=True, bottom=True)
fp.fig.legends[0].set_title('Solver')

path = 'charts'
if len(sys.argv) > 1:
    path = sys.argv[1]
plt.savefig('{}/chart_fig7.pdf'.format(path), bbox_inches='tight')
# plt.show()
