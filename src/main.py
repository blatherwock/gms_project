"""
This file contains the main logic for training and evaluating a graphical
model that predicts New York Citi Bikes future locations.
"""

import utils
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt

out_folder = os.path.join(os.path.split(__file__)[0], "..", "out")

def process_trips(trips_df):
    print(trips_df.info())
    print(trips_df.dtypes)
    print(trips_df.dtypes)
    print(trips_df.head())
    print(trips_df.tail())
    print("min bikeid: {}".format(trips_df['bikeid'].min()))
    print("max bikeid: {}".format(trips_df['bikeid'].max()))
    print("min start station id: {}".format(trips_df['start station id'].min()))
    print("max start station id: {}".format(trips_df['start station id'].max()))
    print("min end station id: {}".format(trips_df['end station id'].min()))
    print("max end station id: {}".format(trips_df['end station id'].max()))

def plot_select_weeks(start_time_matrix, time_idx):
    print("Plotting total trips")

    total_start_trips = np.sum(start_time_matrix, axis=0)
    plt.plot(total_start_trips.A.flatten())
    plt.title("Total trips in 30 minute intervals from 2013 - 2015")
    plt.savefig("{}/total_trips.png".format(out_folder), format="png")
    plt.clf()



def main():
    # Ensure all data has been downloaded and processed
    #utils.download_trips_dataset()
    #for y in utils.YEARS:
    #   utils.load_trips_dataframe(y)
    #   process_trips(trips_df)

    start_time_matrix, station_idx, time_idx = utils.load_start_time_matrix()
    stop_time_matrix, _, _ = utils.load_stop_time_matrix()

    plot_select_weeks(start_time_matrix, time_idx)



if __name__ == '__main__':
    main()
