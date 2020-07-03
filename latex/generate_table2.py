# Table 1
import json
import pandas as pd

from docrec.metrics import accuracy


template = """
\\begin{table*}[htb]
   \\centering
   \\caption{Full reconstruction performance (original compatibility methods + our ATSP-based solver): $Acc_\pi \pm \sigma $ (\\%%).}
   \\label{tab:rec}
   \\begin{adjustbox}{max width=\\textwidth}
      \\begin{tabular}{lrrrrrr}
         \\toprule
         \multirow{3}{*}{\\textbf{Method}} & \multicolumn{2}{c}{D1 $\cup$ D2} & \multicolumn{2}{c}{D1} & \multicolumn{2}{c}{D2}\\\\
         \cmidrule(l){2-3}  \cmidrule(l){4-5}  \cmidrule(l){6-7}
         & \multicolumn{1}{c}{KBH} & \multicolumn{1}{c}{Concorde} & \multicolumn{1}{c}{KBH} & \multicolumn{1}{c}{Concorde} & \multicolumn{1}{c}{KBH} & \multicolumn{1}{c}{Concorde} \\\\
         \\midrule
%s
         \\bottomrule
      \\end{tabular}
    \\end{adjustbox}
\\end{table*}
"""

algorithms = ['proposed_squeezenet', 'proposed_mobilenet', 'andalo', 'morandell', 'balme', 'sleit', 'marques']
proposed_squeezenet = json.load(open('results/proposed_squeezenet.json', 'r'))
proposed_mobilenet = json.load(open('results/proposed_mobilenet.json', 'r'))
fair = json.load(open('results/fair.json', 'r'))

records = []
concorde = 0
kbh = 0
for entry in proposed_squeezenet + proposed_mobilenet + fair:
    algorithm, solver, doc, acc, qc = entry[: 5]
    dataset = doc.split('/')[1]
    if solver == 'Concorde':
        concorde = acc
    else:
        kbh = acc
        records.append([algorithm, dataset,  concorde, kbh])

df = pd.DataFrame.from_records(records, columns=('algorithm', 'dataset', 'Concorde', 'KBH'))
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
df_avg_std_D1 = df[df['dataset']=='D1'].groupby(['algorithm']).agg(['mean', 'std'])
df_avg_std_D2 = df[df['dataset']=='D2'].groupby(['algorithm']).agg(['mean', 'std'])
df_table = 100 * pd.concat([df_avg_std, df_avg_std_D1, df_avg_std_D2], axis=1).reindex(algorithms)
print(df_table)

body = ''
for row in df_table.iterrows():
    algorithm = row[0]
    num_str = '\\textbf{{{:.2f} $\pm$ {:.2f}}}' if algorithm.startswith('proposed') else '{:.2f} $\pm$ {:.2f}'
    avg_conc, std_conc, avg_D1_conc, std_D1_conc, avg_D2_conc, std_D2_conc, avg_kbh, std_kbh, avg_D1_kbh, std_D1_kbh, avg_D2_kbh, std_D2_kbh= row[1]
    body += '         {} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & '.format(
        map_algorithm_legend[algorithm], avg_kbh, std_kbh, avg_D1_kbh, std_D1_kbh, avg_D2_kbh, std_D2_kbh
    )
    body += '{:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f}\\\\{}'.format(
        avg_conc, std_conc, avg_D1_conc, std_D1_conc, avg_D2_conc, std_D2_conc,
        '\n' if algorithm != algorithms[-1] else ''
    )
# change path to save the latex table in other place
table = template % body
print(table)
open('latex/table2.tex', 'w').write(table)