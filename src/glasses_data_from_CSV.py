import pandas as pd
from glasses_utils import *
import os

plot = True
save = True

directory = os.path.join('..", "dataset')

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath) and not os.path.isfile(
        os.path.join(
            directory,
            "fast",
            os.path.basename(filepath)[:-4] + "_fast.csv"
        )
    ):
        print(filepath)

        fast_df, slow_df, interval = load_data(filepath)

        if save:
            fast_df.to_csv(os.path.join(
                directory,
                "fast",
                filename[:-4] + '_fast.csv'
            ))
            slow_df.to_csv(
                os.path.join(
                    directory
                    "slow",
                    filename[:-4] + '_slow.csv'
                )
            )
            with open(os.path.join(
                directory,
                "intervals",
                filename[:-4] + '_interval.txt'
            ), "w") as f:
                f.write(interval)

        if plot: 
            plot_data(fast_df, slow_df, filepath)
