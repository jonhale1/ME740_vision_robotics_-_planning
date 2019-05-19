import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

import numpy as np
import math

df = pd.read_csv('basic_pso_data.txt', header=0, delimiter=',')
r, c = df.shape

# print(tabulate(df.groupby(['Inertia']).mean(), headers='keys', tablefmt='psql'))
print(df.groupby(['Inertia', 'C1', 'C2']).mean())

df.groupby(['Inertia', 'C1', 'C2']).mean().unstack().plot(kind='bar', fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.title('Steps to Convergence vs. Inertia, C1 & C2 Coefficients',fontweight='bold')
plt.ylabel('Steps to Convergence\n(200 = local optimum)')
plt.show()