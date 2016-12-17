import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

import utils

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
    avg_weekly_flow = np.matrix(avg_weekly_flow)
    return lambda idx: avg_weekly_flow[:, idx % (48*7)]
