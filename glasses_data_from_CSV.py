import pandas as pd
from glasses_utils import *
import os

plot = True
save = True

directory = 'df_dataset'

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        print(filepath)

        fast_df, slow_df, interval = load_data(filepath)

        if save:
            fast_df.to_csv(f'{directory}/fast/' + filename[:-4] + '_fast.csv')
            slow_df.to_csv(f'{directory}/slow/' + filename[:-4] + '_slow.csv')
            with open(f'{directory}/intervals/' + filename[:-4] + '_interval.txt', 'w') as f:
                f.write(interval)

        if plot: 
            plot_data(fast_df, slow_df, filepath)
