"""
This file contains the main logic for training and evaluating a graphical
model that predicts New York Citi Bikes future locations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import os
import math

import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import utils
import gmm

plt.style.use('ggplot')

out_folder = os.path.join(os.path.split(__file__)[0], "..", "out")

def savefig(file_name):
    plt.savefig(os.path.join(out_folder, file_name))
    plt.clf()

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


def plot_avg_week_for_stations(avg,
                               station_idx,
                               time_at_idx,
                               station_ids,
                               plot_title,
                               file_name):
    print("Plotting average weeks for stations")

    fig = plt.figure()
    ax = plt.subplot(111)

    plt.title(plot_title)
    plt.ylabel('Number of trips')
    plt.xlabel('Time bucket')

    x_axis = [time_at_idx(i) for i in range(0, 48*7)]
    for station_id in station_ids:
        print("\r\tPlotting average week for station {}".format(station_id), end="")
        ax.plot(x_axis, avg[station_idx[station_id],:], linestyle="solid", alpha=0.8, label=station_id)
        print("\r" + " "*80 + "\r", end="")
    xticks = [ x for x in x_axis if x.minute == 0 and x.hour in [0,6,12,18] ]
    xticklabels = [ x.strftime("%a") if x.hour == 0 else x.hour if x.hour in [12] else "" for x in xticks ]
    plt.xticks(xticks, xticklabels, rotation=70)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(loc="upper right", bbox_to_anchor=(1.25,1))
    savefig(file_name)


def plot_total_start_trips(start_time_matrix, time_idx):
    print("Plotting total trips")

    def plot_total_over_year(interval, interval_name, file_name):
        n_stations, total_buckets = start_time_matrix.shape
        if start_time_matrix.shape[1] % interval != 0:
            end_index = math.floor(total_buckets / interval) * interval
            temp_trips = start_time_matrix[:, :end_index].todense().A
        else:
            temp_trips = start_time_matrix.todense().A
        temp_trips = temp_trips.reshape((n_stations, -1, interval))
        start_time_days = np.sum(temp_trips, axis=2)
        total_start_trips = np.sum(start_time_days, axis=0)
        plt.plot(total_start_trips)
        plt.title("Total trips in {} intervals from 2013 - 2015".format(interval_name))
        plt.ylabel("# of trips / {}".format(interval_name))
        # Take the original month tickmarks and divide them by interval to find the appropriate
        # marks for the transformed data
        tickmarks = np.array(utils.month_indices()) / interval
        ticklabels = ['']*len(tickmarks)
        ticklabels[0] = "2013"
        ticklabels[6] = "2014"
        ticklabels[18] = "2015"
        ticklabels[30] = "2016"
        plt.xticks(tickmarks, ticklabels, rotation=70)
        savefig(file_name)

    # Plot total started trips for each bucket, day, and week
    plot_total_over_year(1, "30 minute", "total_trips_30_min_buckets.pdf")
    plot_total_over_year(48, "day", "total_trips_days.pdf")
    plot_total_over_year(48 * 7, "week", "total_trips_weeks.pdf")

    # Plot total started trips for each 30 minute bucket shown as a single day
    # ie the total number of trips that have been started between 00:00 and 00:30, etc
    n_stations, total_buckets = start_time_matrix.shape
    temp_trips = start_time_matrix.todense().A
    temp_trips = temp_trips.reshape((n_stations, -1, 48))
    start_time_buckets = np.sum(np.sum(temp_trips, axis=1), axis=0)
    plt.plot(start_time_buckets)
    plt.title("Total trips for each 30 min window")
    plt.ylabel("Total # of trips / 30 minute window")
    # Take the original month tickmarks and divide them by 48 to find the appropriate
    # marks for the transformed data
    tickmarks = np.array(range(24)) * 2
    ticklabels = [ "{:0>2}:00".format(math.floor(i)) for i in range(48) ]
    plt.xticks(tickmarks, ticklabels, rotation=70)
    savefig("total_trips_30_min_bucket_over_day.pdf")


def plot_normalized_avg_week_for_stations(avg,
                                          station_idx,
                                          time_at_idx,
                                          station_ids,
                                          plot_title,
                                          file_name):
    print("Plotting normalized average weeks for stations")
    # Normalize the stations
    maxes = np.max(np.abs(avg), axis=1)
    maxes = np.repeat(maxes, avg.shape[1]).reshape(avg.shape)
    avg = np.divide(avg, maxes)

    plt.title(plot_title)
    plt.ylabel('Number of trips')
    plt.xlabel('Time bucket')

    x_axis = [time_at_idx(i) for i in range(0, 48*7)]
    for station_id in station_ids:
        print("\r\tPlotting average week for station {}".format(station_id), end="")
        plt.plot(x_axis, avg[station_idx[station_id],:], linestyle="solid", alpha=0.8, label=station_id)
        print("\r" + " "*80 + "\r", end="")
    xticks = [ x for x in x_axis if x.minute == 0 and x.hour in [0,6,12,18] ]
    xticklabels = [ x.strftime("%a") if x.hour == 0 else x.hour if x.hour in [12] else "" for x in xticks ]
    plt.xticks(xticks, xticklabels, rotation=70)
    plt.legend(loc="upper right")
    savefig(file_name)


def plot_tsne(avg, inverse_station, clusters=None, plot_title="t-SNE", file_name="t-SNE.pdf"):
    print("Plotting t-SNE...")
    model = TSNE(n_components=2, random_state=0)
    avg_matrix_2d = model.fit_transform(avg)
    X, Y = avg_matrix_2d[:,0], avg_matrix_2d[:,1]
    plt.title(plot_title)
    plt.scatter(X, Y, c=clusters, cmap=cm.gist_rainbow)
    for i, xy in enumerate(zip(X, Y)):
        plt.annotate("{}".format(inverse_station[i] if i in inverse_station else ""), xy=xy, textcoords='data', fontsize=2)
    savefig(file_name)


def get_weekly_mean(complete_matrix):
    # -5*48 to exclude last 5 days, to end on Sunday at 23:59
    mat = complete_matrix[:,:-5*48].todense().A
    n_stations, total_buckets = mat.shape
    mat = mat.reshape((n_stations, -1, 48*7))
    return np.mean(mat, axis=1)


def main():
    # Ensure all data has been downloaded and processed
    #utils.download_trips_dataset()
    #for y in utils.YEARS:
    #   utils.load_trips_dataframe(y)
    #   process_trips(trips_df)

    start_time_matrix, station_idx, time_idx, time_at_idx = utils.load_start_time_matrix()
    stop_time_matrix, _, _ = utils.load_stop_time_matrix()
    start_time_matrix = start_time_matrix.astype(np.int16)
    stop_time_matrix = stop_time_matrix.astype(np.int16)
    inverse_station = { v: k for k, v in station_idx.items() }

    flux_matrix = stop_time_matrix - start_time_matrix

    # plot_tsne(get_weekly_mean(flux_matrix), inverse_station)

    # plot_avg_week_for_stations(get_weekly_mean(start_time_matrix), station_idx, time_at_idx, [360], "Number of trips started at station over week", "avg_week_start_time.pdf")
    # plot_avg_week_for_stations(get_weekly_mean(stop_time_matrix), station_idx, time_at_idx, [360], "Number of trips stopped at station over week", "avg_week_stop_time.pdf")
    # Some interesting stations: 3412, 3324, 3285, 3286, 3153, 360, 195, 2023, 3095, 432, 511, 438
    # plot_avg_week_for_stations(get_weekly_mean(flux_matrix), station_idx, time_at_idx, [360, 195, 146, 432, 161, 497, 517], "Net change in bikes at station over week","avg_week_flow_time.pdf")
    # plot_total_start_trips(start_time_matrix, time_idx)

    # plot_normalized_avg_week_for_stations(get_weekly_mean(flux_matrix), station_idx, time_at_idx, [360, 195, 497, 146, 161], 
    #     "Net change in bikes at station over week (normalized)","normalized_avg_week_flow_time.pdf")

    flux_matrix_weekly_mean = get_weekly_mean(flux_matrix)
    model = TSNE(n_components=2, random_state=0)
    avg_matrix_2d = model.fit_transform(flux_matrix_weekly_mean)
    avg_matrix_2d = avg_matrix_2d[:10,:] / 40.0
    X, Y = avg_matrix_2d[:,0], avg_matrix_2d[:,1]
    plt.title("t-SNE")
    plt.scatter(X, Y)
    for i, xy in enumerate(zip(X, Y)):
        plt.annotate("{}".format(inverse_station[i] if i in inverse_station else ""), xy=xy, textcoords='data', fontsize=2)
    savefig("t-SNE.pdf")
    # pdb.set_trace()

    clusters = gmm.gmm(avg_matrix_2d, K=4, D=2)
    # clusters = [0, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4,
    #    0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4, 4, 0,
    #    0, 0, 0, 4, 0, 0, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0,
    #    0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4,
    #    4, 4, 0, 4, 0, 4, 0, 4, 4, 4, 4, 4, 0, 0, 4, 0, 2, 4, 0, 0, 0, 0, 4,
    #    4, 0, 4, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 4, 4, 4, 4, 0, 4, 0, 0,
    #    0, 4, 0, 0, 4, 0, 4, 0, 0, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0,
    #    4, 4, 0, 0, 4, 0, 4, 4, 4, 4, 0, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 4,
    #    0, 0, 0, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 4, 0, 0, 4, 4, 0, 0, 4, 0,
    #    2, 4, 4, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4,
    #    4, 4, 0, 0, 0, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4,
    #    0, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 0, 4, 4, 4, 0, 4, 0, 4, 4, 0, 0,
    #    0, 0, 0, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 0, 4, 4, 4,
    #    4, 4, 4, 4, 4, 0, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    # clusters = [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # clusters = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    print(clusters)
    pdb.set_trace()
    plt.title("t-SNE")
    plt.scatter(X, Y, c=clusters, cmap=cm.gist_rainbow)
    for i, xy in enumerate(zip(X, Y)):
        plt.annotate("{}".format(inverse_station[i] if i in inverse_station else ""), xy=xy, textcoords='data', fontsize=2)
    savefig("Clustered_t-SNE.pdf")

    # plot_tsne(flux_matrix_weekly_mean, inverse_station, clusters=clusters)
    

if __name__ == '__main__':
    main()
