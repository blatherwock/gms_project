import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

import utils
import pdb

def fit_seasonal_trend(bucketed_totals):
    def fit(X, Y):
        gp_kernel = RBF(300) + ExpSineSquared(1.0, 52.0, periodicity_bounds=(1e-2, 1e10)) + WhiteKernel(1e-1)
        model = GaussianProcessRegressor(kernel=gp_kernel)
        model.fit(X, Y)
        return model
    def predict_fn(model, scaling_factor):
        return lambda x: model.predict(x) * scaling_factor

    n_buckets = bucketed_totals.shape[0]
    scaling_factor = 100000
    X = np.array(range(n_buckets)).reshape((n_buckets, 1))
    Y = bucketed_totals.reshape((n_buckets, 1)) / scaling_factor
    return predict_fn(fit(X, Y), scaling_factor)


def train_avg_week_predictor(flow_matrix):
    avg_weekly_flow = utils.get_station_agg_trips_over_week(flow_matrix, np.mean)
    avg_weekly_flow = np.matrix(avg_weekly_flow).A
    return lambda idx: avg_weekly_flow[:, idx % (48*7)]


def train_seasonal_avg_week_predictor(start_time_matrix, flow_matrix):
    avg_weekly_flow = utils.get_station_agg_trips_over_week(flow_matrix, np.mean)
    # note that the x values for the seasonal predictor are in weeks, not 30 min buckets
    total_start_trips = utils.get_total_weekly_trips(start_time_matrix)
    average_trip_volume = np.mean(total_start_trips)
    seasonal_predictor = fit_seasonal_trend(total_start_trips)
    avg_predictor = train_avg_week_predictor(flow_matrix)

    def predict_fn(idx):
        idx_week = np.floor(idx / utils.INTERVAL_WEEKLY).astype(np.int32)
        idx_week = idx_week.reshape((-1,1))
        seasonal_multiplier = (seasonal_predictor(idx_week) / average_trip_volume).T

        prediction = avg_predictor(idx)
        seasonal_multiplier = np.tile(seasonal_multiplier, prediction.shape[0]).reshape(prediction.shape)

        return np.multiply(prediction, seasonal_multiplier)

    return predict_fn


def train_cluster_based_predictor(start_time_matrix, flow_matrix, cluster_assignments, cluster_mus):
    avg_weekly_flow = utils.get_station_agg_trips_over_week(flow_matrix, np.mean)
    # note that the x values for the seasonal predictor are in weeks, not 30 min buckets
    total_start_trips = utils.get_total_weekly_trips(start_time_matrix)
    average_trip_volume = np.mean(total_start_trips)
    seasonal_predictor = fit_seasonal_trend(total_start_trips)
    avg_predictor = train_avg_week_predictor(flow_matrix)

    def predict_fn(idx):
        idx_week = np.floor(idx / utils.INTERVAL_WEEKLY).astype(np.int32)
        idx_week = idx_week.reshape((-1,1))
        seasonal_multiplier = (seasonal_predictor(idx_week) / average_trip_volume).T

        prediction = cluster_mus[cluster_assignments][:, idx % (utils.INTERVAL_WEEKLY)]
        seasonal_multiplier = np.tile(seasonal_multiplier, prediction.shape[0]).reshape(prediction.shape)

        return np.multiply(prediction, seasonal_multiplier)

    return predict_fn
