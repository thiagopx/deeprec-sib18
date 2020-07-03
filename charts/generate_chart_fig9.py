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

colors = sns.color_palette('deep')
order = [5, 0, 1, 9, 3, 8, 2, 4, 6, 7]
pallete = [colors[i] for i in order]
sns.set_palette(pallete)

algorithms = ['proposed_squeezenet', 'proposed_mobilenet', 'marques', 'morandell', 'andalo', 'balme', 'sleit']
proposed_squeezenet = json.load(open('results/proposed_squeezenet.json', 'r'))
proposed_mobilenet = json.load(open('results/proposed_mobilenet.json', 'r'))
fair = json.load(open('results/fair.json', 'r'))
unfair = json.load(open('results/unfair.json', 'r'))

records_proposed = []
records_fair = []
records_unfair = []

for records, results in zip([records_proposed, records_fair, records_unfair], [proposed_squeezenet + proposed_mobilenet, fair, unfair]):
    for entry in results:
        algorithm, solver, doc, acc = entry[: 4]
        if solver == 'Concorde':
            dataset = doc.split('/')[1]
            records.append([algorithm, dataset, 100 * acc])

# Chart Fig9a
map_algorithm_legend = dict(
    proposed_squeezenet='\\textbf{Proposed-SN}',
    proposed_mobilenet='\\textbf{Proposed-MN}',
    andalo='Andaló',
    morandell='Morandell',
    balme='Balme',
    sleit='Sleit',
    marques='Marques'
)
legends = [map_algorithm_legend[algorithm] for algorithm in algorithms]
df = pd.DataFrame.from_records(records_proposed + records_fair, columns=('algorithm', 'dataset', 'accuracy'))
df_union = df.copy()
df_union['dataset'] = 'D1 $\cup$ D2'
df = pd.concat([df_union, df])
df['legend'] = df['algorithm'].map(map_algorithm_legend)
fp = sns.catplot(
    x='dataset', y='accuracy', data=df, hue='legend', kind='box',
    hue_order=legends, height=3, aspect=2.5,
    margin_titles=True, fliersize=1, width=0.8, linewidth=1.5,
    legend=False,
)
fp.despine(left=True, bottom=True)
path = 'charts'
if len(sys.argv) > 1:
    path = sys.argv[1]
plt.savefig('{}/chart_fig9a.pdf'.format(path), bbox_inches='tight')

# Chart Fig9a
df = pd.DataFrame.from_records(records_proposed + records_unfair, columns=('algorithm', 'dataset', 'accuracy'))
df_union = df.copy()
df_union['dataset'] = 'D1 $\cup$ D2'
df = pd.concat([df_union, df])
df['legend'] = df['algorithm'].map(map_algorithm_legend)
fp = sns.catplot(
    x='dataset', y='accuracy', data=df, hue='legend', kind='box',
    hue_order=legends, height=3, aspect=2.5,
    margin_titles=True, fliersize=1, width=0.8, linewidth=1.5,
    legend=False,
)
fp.despine(left=True, bottom=True)
plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.3), ncol=3)
path = 'charts'
if len(sys.argv) > 1:
    path = sys.argv[1]
plt.savefig('{}/chart_fig9b.pdf'.format(path), bbox_inches='tight')

