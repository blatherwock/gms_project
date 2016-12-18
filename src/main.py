"""
This file contains the main logic for training and evaluating a graphical
model that predicts New York Citi Bikes future locations.
"""

import pdb
import os
import math

import numpy as np
from numpy.random import randint
from sklearn.manifold import TSNE
from scipy.misc import imread

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import utils
import gmm
import prediction

plt.style.use('ggplot')

out_folder = os.path.join(os.path.split(__file__)[0], "..", "out")
cluster_colors = np.array(list("rgbcyk"))


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


def plot_avg_week_for_stations(start_time_matrix,
                               station_idx,
                               time_at_idx,
                               station_ids=None,
                               plot_title="weekly behavior",
                               file_name="temp.pdf",
                               normalize=False,
                               round=False):
    print("Plotting normalized average weeks for stations")
    if (start_time_matrix.shape[1] > utils.INTERVAL_WEEKLY):
        avg = utils.get_station_agg_trips_over_week(start_time_matrix, np.mean)
    else:
        avg = start_time_matrix
    if normalize:
        avg = utils.normalize(avg)
    if round:
        avg = utils.round(avg, 4)

    plt.title(plot_title)
    plt.ylabel('Number of trips')
    plt.xlabel('Time bucket')

    x_axis = [time_at_idx(i) for i in range(0, 48*7)]
    if station_ids is not None:
        for station_id in station_ids:
            plt.plot(x_axis, avg[station_idx[station_id],:], linestyle="solid", alpha=0.8, label=station_id)
    else:
        for i in range(avg.shape[0]):
            plt.plot(x_axis, avg[i], linestyle="solid", alpha=0.03, color="k")
            plt.ylim(-50,40)

    xticks = [ x for x in x_axis if x.minute == 0 and x.hour in [0,6,12,18] ]
    xticklabels = [ x.strftime("%a") if x.hour == 0 else x.hour if x.hour in [12] else "" for x in xticks ]
    plt.xticks(xticks, xticklabels, rotation=70)
    if station_ids is not None:
        plt.legend(loc="upper right")
    savefig(file_name)


def plot_total_start_trips(start_time_matrix, time_idx):
    print("Plotting total trips")

    def plot_total_over_year(interval, interval_name, file_name):
        plt.plot(utils.get_agg_trips_by_interval(start_time_matrix, interval))
        plt.title("Total trips in {} intervals from 2013 - 2015".format(interval_name))
        plt.ylabel("# of trips / {}".format(interval_name))
        plt.xticks(*utils.year_labels(interval), rotation=70)
        savefig(file_name)

    # Plot total started trips for each bucket, day, and week
    plot_total_over_year(1, "30 minute", "total_trips_30_min_buckets.pdf")
    plot_total_over_year(utils.INTERVAL_DAILY, "day", "total_trips_days.pdf")
    plot_total_over_year(utils.INTERVAL_WEEKLY, "week", "total_trips_weeks.pdf")

    # Plot total started trips for each 30 minute bucket shown as a single day
    # ie the total number of trips that have been started between 00:00 and 00:30, etc
    def plot_total_over_day(aggregator, name, file_name):
        plt.plot(utils.get_agg_trips_over_day(start_time_matrix, aggregator))
        plt.title("{} trips for each 30 min window".format(name))
        plt.ylabel("{} # of trips / 30 minute window".format(name))
        # Take the original month tickmarks and divide them by 48 to find the appropriate
        # marks for the transformed data
        tickmarks = np.array(range(24)) * 2
        ticklabels = [ "{:0>2}:00".format(math.floor(i)) if i % 2 == 0 else "" for i in range(48) ]
        plt.xticks(tickmarks, ticklabels, rotation=70)
        savefig(file_name)

    plot_total_over_day(np.sum, "Total", "total_trips_30_min_bucket_over_day.pdf")
    plot_total_over_day(np.mean, "Avg", "avg_trips_30_min_bucket_over_day.pdf")


def plot_predicted_total_start_trips(start_time_matrix):
    interval_name = "week"
    total_start_trips = utils.get_total_weekly_trips(start_time_matrix)
    buckets = total_start_trips.shape[0]
    average_trip_volume = np.mean(total_start_trips)
    plt.plot(range(buckets), [average_trip_volume]*buckets, linestyle='dotted', color='k')

    predict_using = 169

    # predict 10 weeks into the future
    prediction_buckets = buckets + 52
    predicted_x_plot = np.linspace(0, prediction_buckets, prediction_buckets*10)[:,None]
    X = np.array(list(range(buckets))).reshape((buckets, 1))

    predictor = prediction.fit_seasonal_trend(total_start_trips[:predict_using])
    predicted_y = predictor(predicted_x_plot)

    plt.plot(predicted_x_plot, predicted_y, color='g', label='Prediction')

    plt.plot(total_start_trips[:predict_using], color='r', label='Total Weekly Trips')
    plt.plot(range(predict_using, buckets), total_start_trips[predict_using:], color='b')

    plt.title("Total trips in {} intervals from 2013 - 2015".format(interval_name))
    plt.ylabel("# of trips / {}".format(interval_name))
    plt.xticks(*utils.year_labels(utils.INTERVAL_WEEKLY), rotation=70)
    plt.legend(loc='upper left')
    savefig('total_weekly_trips_prediction.pdf')

def plot_predicted_flow_baseline(flow_matrix):
    # TODO: Figure out why this doesn't look the way we expect it to look
    avg_week_predictor = prediction.train_avg_week_predictor(flow_matrix)
    idx = np.array(range(flow_matrix.shape[1]))
    errors = np.mean(np.abs(avg_week_predictor(idx) - flow_matrix), axis=0)
    errors = errors.A.flatten()
    plt.plot(errors)
    plt.title("Total error in baseline predictor")
    plt.ylabel("mean error")
    plt.xticks(*utils.year_labels(), rotation=70)
    savefig("baseline_predictor_errors.pdf")

def plot_map(cluster_assignments,
             station_info,
             station_idx,
             inverse_station,
             plot_title="Bike Locations",
             file_name="station_map.pdf",
             add_labels=False):
    print("Plotting NYC map")
    locations = np.zeros((len(station_idx), 2))
    skipped_locs = []
    for station_id, idx in station_idx.items():
        if station_id in station_info:
            locations[idx, 0] = station_info[station_id]['longitude']
            locations[idx, 1] = station_info[station_id]['latitude']
        else:
            # One of the corners of our data
            locations[idx, 0] = -74.01713445
            locations[idx, 1] = 40.804213
            skipped_locs.append(station_id)
    print("Couldn't find infomation about stations: {}".format(skipped_locs))

    img = imread('./src/nyc.png')
    plt.imshow(img, zorder=0, extent=[-74.10, -73.85, 40.65, 40.85])

    X = locations[:,0]
    Y = locations[:,1]
    plt.title(plot_title)
    plt.scatter(X, Y, c=cluster_colors[cluster_assignments].tolist(), zorder=1)
    plt.axis([-74.10, -73.85, 40.65, 40.85])

    if add_labels:
        for i in range(0,locations.shape[0]):
            xy = (locations[i,0], locations[i,1])
            station_id = inverse_station[i]
            if station_id in station_info:
                name = "{}".format(station_info[station_id]['stationName'] if i in inverse_station else "")
                plt.annotate(name, xy=xy, textcoords='data', fontsize=2)
    savefig(file_name)

def plot_cluster_means(mus,
                       time_at_idx,
                       plot_title="Cluster means (week)",
                       file_name="cluster_means.pdf"):
    plt.title(plot_title)
    plt.ylabel('Number of trips')
    plt.xlabel('Time bucket')

    x_axis = [time_at_idx(i) for i in range(0, 48*7)]
    for i in range(mus.shape[0]):
        plt.plot(x_axis, mus[i], linestyle="solid", alpha=0.8, label="Cluster {}".format(i), color=cluster_colors[i])

    xticks = [ x for x in x_axis if x.minute == 0 and x.hour in [0,6,12,18] ]
    xticklabels = [ x.strftime("%a") if x.hour == 0 else x.hour if x.hour in [12] else "" for x in xticks ]
    plt.xticks(xticks, xticklabels, rotation=70)
    plt.legend(loc="upper right")
    savefig(file_name)


def plot_clustered_stations_2d(avg_weekly_flow,
                               cluster_assignments,
                               means,
                               inverse_station,
                               plot_title="Clustered stations (t-SNE representation)",
                               file_name="clustered_stations.pdf"):
    print("Plotting station clusters")
    K = means.shape[0]
    X = np.vstack([avg_weekly_flow, means])

    # t-SNE for X
    model = TSNE(n_components=2, random_state=0)
    X_tsne = model.fit_transform(X)

    # Plot resulting clusters in 2d
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(plot_title)
    ax.scatter(X_tsne[:-K, 0], X_tsne[:-K, 1], color=cluster_colors[cluster_assignments].tolist(), s=10, alpha=0.7)
    for i, xy in enumerate(zip(X_tsne[:-K, 0], X_tsne[:-K, 1])):
        plt.annotate("{}".format(inverse_station[i] if i in inverse_station else ""), xy=xy, textcoords='data', fontsize=2)
    for k in range(K):
        mu = X_tsne[-K+k, :]
        ax.scatter(mu[0], mu[1], s=50, color=cluster_colors[k], marker="+")
    savefig(file_name)


def main():
    # Ensure all data has been downloaded and processed
    #utils.download_trips_dataset()
    #for y in utils.YEARS:
    #   utils.load_trips_dataframe(y)
    #   process_trips(trips_df)

    np.random.seed(1)

    station_info = utils.load_station_info()
    start_time_matrix, station_idx, time_idx, time_at_idx = utils.load_start_time_matrix()
    stop_time_matrix, _, _ = utils.load_stop_time_matrix()
    start_time_matrix = start_time_matrix.astype(np.int16)
    stop_time_matrix = stop_time_matrix.astype(np.int16)
    inverse_station = { v: k for k, v in station_idx.items() }
    flow_matrix = stop_time_matrix-start_time_matrix

    plot_avg_week_for_stations(start_time_matrix, station_idx, time_at_idx, [360], 
        "Number of trips started at station over week", "avg_week_start_time.pdf")
    plot_avg_week_for_stations(stop_time_matrix, station_idx, time_at_idx, [360], 
        "Number of trips stopped at station over week", "avg_week_stop_time.pdf")
    plot_avg_week_for_stations(flow_matrix, station_idx, time_at_idx, [360, 195, 146, 432, 161, 497, 517], 
        "Net change in bikes at station over week","avg_week_flow_time.pdf")

    plot_total_start_trips(start_time_matrix, time_idx)

    # Some interesting stations: 3412, 3324, 3285, 3286, 3153, 360, 195, 2023, 3095, 432, 511, 438
    plot_avg_week_for_stations(flow_matrix, station_idx, time_at_idx, [360, 195, 497, 146, 161], 
        "Net change in bikes at station over week (normalized)","normalized_avg_week_flow_time.pdf", True)
    plot_avg_week_for_stations(flow_matrix, station_idx, time_at_idx, [360, 195, 497, 146, 161], 
        "Net change in bikes at station over week (normalized, rounded)","normalized_round_avg_week_flow_time.pdf", True, True)
    plot_avg_week_for_stations(flow_matrix, station_idx, time_at_idx, None, 
        "Net change in bikes at station over week (normalized)","normalized_all_avg_week_flow_time.pdf", True)
    plot_avg_week_for_stations(flow_matrix, station_idx, time_at_idx, None, 
        "Net change in bikes at station over week ","all_avg_week_flow_time.pdf")
    plot_avg_week_for_stations(flow_matrix, station_idx, time_at_idx, None, 
        "Net change in bikes at station over week (normalized, rounded)","normalized_round_all_avg_week_flow_time.pdf", True, True)

    # Cluster stations
    print("Clustering stations")
    avg_weekly_flow = utils.get_station_agg_trips_over_week(flow_matrix, np.mean)
    cluster_assignments, means, ppc = gmm.gmm(avg_weekly_flow, K=3, posterior_predictive_check=True)

    # Plot the posterior predictive check
    random_indices = randint(0, flow_matrix.shape[0], size=150)
    plot_avg_week_for_stations(ppc[0][random_indices], station_idx, time_at_idx, None, "Posterior Predictive Check on average flow", "post_pred_check.pdf")
    plot_avg_week_for_stations(flow_matrix[random_indices], station_idx, time_at_idx, None, "Average flow for 150 random stations", "post_pred_check_orig.pdf")

    # Plot clustered stations in 2d
    plot_clustered_stations_2d(avg_weekly_flow, cluster_assignments, means, inverse_station)

    # Plot weekly graph for mean of each cluster
    plot_cluster_means(means, time_at_idx)

    # Plot clustered stations on map
    plot_map(cluster_assignments, station_info, station_idx, inverse_station)

    # Predictions
    plot_predicted_total_start_trips(start_time_matrix)
    plot_predicted_flow_baseline(flow_matrix)

    

if __name__ == '__main__':
    main()
