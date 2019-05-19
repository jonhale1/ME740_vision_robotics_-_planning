import pandas as pd
import numpy as np
import math
from tabulate import tabulate

df = pd.read_csv('data.txt', header=0, delimiter=',')
r, c = df.shape

def clean_data(df):
    """Cleans the data frame and preps for analysis"""
    for i in range(0, r-1):
        if df.at[i, 'Found Step'] == 0:
            df.at[i, 'Steps Saved'] = df.at[i, 'Found Step'] - df.at[i, 'Time Elapsed']


def time_saved(df):
    """Performs data analysis"""
    time_saved = pd.DataFrame((df['Steps Saved']*12)/60)
    time_saved.rename(index=str, columns={'Steps Saved':'Time Saved (hrs)'}, inplace=True)
    return time_saved


clean_data(df)
time_saved = time_saved(df)

print(tabulate(df, headers='keys', tablefmt='psql'))
print('\n')
print(tabulate(df.groupby(['Inertia']).mean(), headers='keys', tablefmt='psql'))
print('\n')
print(tabulate(time_saved, headers='keys', tablefmt='psql'))
print('Average {}essentially dampens each particles '.format(time_saved.mean()))
print('\n')

