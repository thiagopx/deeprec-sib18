# Table 1
import json
import pandas as pd

from docrec.metrics import Qc

template = """
\\begin{table}[b]
   \\centering
   \\caption{Performance of the compatibility scoring methods: $Q_{\mathbf{C}} \pm \sigma$ (\\%%).}
   \\label{tab:matrix}
   \\begin{tabular}{lrrr}
   \\toprule
      \\textbf{Method} & \multicolumn{1}{c}{D1 $\cup$ D2} & \multicolumn{1}{c}{D1} & \multicolumn{1}{c}{D2}\\\\
      \\midrule
%s
      \\bottomrule
   \\end{tabular}
\\end{table}
"""

algorithms = ['proposed_squeezenet', 'proposed_mobilenet', 'andalo', 'morandell', 'balme', 'sleit', 'marques']
proposed_squeezenet = json.load(open('results/proposed_squeezenet.json', 'r'))
proposed_mobilenet = json.load(open('results/proposed_mobilenet.json', 'r'))
fair = json.load(open('results/fair.json', 'r'))
unfair = json.load(open('results/unfair.json', 'r'))

records = []
for entry in proposed_squeezenet + proposed_mobilenet + unfair:
    algorithm, solver, doc, accuracy, qc = entry[: 5]
    # only to filter duplicated entries
    if solver == 'Concorde':
        dataset = doc.split('/')[1]
        records.append([algorithm, dataset, qc])

df = pd.DataFrame.from_records(records, columns=('algorithm', 'dataset', 'Qc'))
map_algorithm_legend = dict(
    proposed_squeezenet='\\textbf{Proposed-SN}',
    proposed_mobilenet='\\textbf{Proposed-MN}',
    andalo='Andaló',
    morandell='Morandell',
    balme='Balme',
    sleit='Sleit',
    marques='Marques'
)

df_avg_std = df.groupby(['algorithm']).agg(['mean', 'std'])
df_avg_std_D1= df[df['dataset']=='D1'].groupby(['algorithm']).agg(['mean', 'std'])
df_avg_std_D2 = df[df['dataset']=='D2'].groupby(['algorithm']).agg(['mean', 'std'])
df_table = 100 * pd.concat([df_avg_std, df_avg_std_D1, df_avg_std_D2], axis=1).reindex(algorithms)

body = ''
for row in df_table.iterrows():
    algorithm = row[0]
    num_str = '\\textbf{{{:.2f} $\pm$ {:.2f}}}' if algorithm.startswith('proposed') else '{:.2f} $\pm$ {:.2f}'
    avg, std, avg_D1, std_D1, avg_D2, std_D2 = row[1]
    body += '      {} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f}\\\\{}'.format(
        map_algorithm_legend[algorithm], avg, std, avg_D1, std_D1, avg_D2, std_D2,
        '\n' if algorithm != algorithms[-1] else ''
    )
table = template % body
print(table)
open('latex/table1.tex', 'w').write(table)