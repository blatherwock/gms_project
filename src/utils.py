"""
This file contains implementation of some needed utility funtions.
"""

import os
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import requests
import pickle
import glob
import math
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import hstack

import pdb

data_dir = os.path.join(os.path.split(__file__)[0], "..", "data")
dataset_dir = os.path.join(data_dir, "trip_histories")

YEARS = ["13", "14", "15", "16"]
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DATA_START = datetime.strptime("2013-07-01 00:00:00", DATE_FORMAT)

# I/O Utils
def download_trips_dataset(force_download=False):
    """Downloads Citi Bike Trip Histories dataset."""
    if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

    if os.path.isdir(dataset_dir) and not force_download:
        print("Dataset trip_histories already exists. Skipping download.")
        return

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    base_url = "https://s3.amazonaws.com/tripdata/"
    zip_file_name_pattern = "\d+-citibike-tripdata.zip"
    root = ET.parse(urllib.request.build_opener().open(base_url)).getroot()
    for child in root:
        if child.tag.endswith("Contents") and len(child) > 0:
            zip_file_name = child[0].text
            if re.match(zip_file_name_pattern, zip_file_name):
                print("Downloading %s..." % zip_file_name)
                zip_file_url = base_url + zip_file_name
                zip_file_path = os.path.join(dataset_dir, zip_file_name)
                urllib.request.urlretrieve(zip_file_url, zip_file_path)
                print("Extracting %s..." % zip_file_name)
                base_file_path = os.path.splitext(zip_file_path)[-2]
                with zipfile.ZipFile(zip_file_path) as zf:
                    for csv_file in zf.namelist():
                        csv_file_path = os.path.join(dataset_dir, csv_file)
                        zf.extract(csv_file, dataset_dir)
                        os.rename(csv_file_path, "%s.csv" % base_file_path)
                os.remove(zip_file_path)

def load_station_info(force_download=False):
    #info_url = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
    info_url = "https://feeds.citibikenyc.com/stations/stations.json"
    station_info_pkl = os.path.join(data_dir, "station_info.pkl")

    if os.path.isfile(station_info_pkl):
        print("{} already exist, skipping download".format(station_info_pkl))
    else:
        info_json = requests.get(info_url).json()
        #station_info_list = info_json['data']['stations']
        station_info_list =info_json['stationBeanList']
        #station_info =  { elem['station_id'] : elem for elem in station_info_list }
        station_info =  { elem['id'] : elem for elem in station_info_list }
        with open(station_info_pkl, 'wb') as pkl:
            pickle.dump(station_info, pkl)

    with open(station_info_pkl, 'rb') as pkl:
        return pickle.load(pkl)

def load_trips_dataframe(year=16):
    """Loads Citi Bike Trips Histories dataset into Pandas' dataframe"""
    trip_histories_pkl = os.path.join(data_dir, "trips_history{}.pkl".format(year))
    if os.path.isfile(trip_histories_pkl):
        print("trip_history{}.pkl already exists. Skipping pickling.".format(year))
    else:
        dataframes = []
        all_files = glob.glob(dataset_dir + "/20{}*.csv".format(year))
        for file in all_files:
            print("Loading {}...".format(file))
            dataframes.append(pd.read_csv(file, usecols=["starttime", "stoptime", "start station id", "end station id", "bikeid"], parse_dates=["starttime", "stoptime"]))
        print("Concatenating all loaded files and pickling result...")
        pd.concat(dataframes).reset_index(drop=True).to_pickle(trip_histories_pkl)
    print("Loading Pickle...", end="")
    temp = pd.read_pickle(trip_histories_pkl)
    print("\rLoading Complete")
    return temp

def time_idx(start_time):
    start_time_obj = datetime.strptime(start_time, DATE_FORMAT) if isinstance(start_time, str) else start_time

    if start_time_obj < DATA_START:
        raise IndexError("start_time is earlier than data start")
    difference = start_time_obj - DATA_START
    # convert the difference (which is stored in days and seconds) into the number of 
    # half hour intervals between the start date and the given start_time
    half_hour_intervals = difference.days * (24 * 2) + difference.seconds / (30 * 60)
    return math.floor(half_hour_intervals)

def time_at_idx(idx):
    delta = timedelta(seconds=idx * (30 * 60))
    return DATA_START + delta

def load_start_time_matrix():
    start_time_matrix_npz = os.path.join(data_dir, "start_time_matrix.npz")
    station_idx = {}
    final_matrix = csr_matrix((0,0), dtype=np.uint8)

    if os.path.isfile(start_time_matrix_npz):
        print("{} already exists. Skipping creation".format(start_time_matrix_npz))
        with np.load(start_time_matrix_npz) as loader:
            # Use [()] to pull the python dict out of the numpy array object
            station_idx = loader['station_idx'][()]
            final_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    else:
        # There are currently 663 stations returned by the station feed.
        # Previous years had less stations, so when constructing the matrix, 
        # this is a overcount so the matrix is large enough (we only return the
        # portion of the matrix that we use)
        MAX_POSSIBLE_STATIONS = 700
        STATION_ID_COL = 3
        START_TIME_COL = 1
        # Map of station id -> idx
        station_idx = {}
        max_station_idx = -1
        start_time_matrix = lil_matrix((MAX_POSSIBLE_STATIONS, 0), dtype=np.uint8)

        for y in YEARS:
            # sample part is if you want to test on a much smaller subset for speed reasons
            trips = load_trips_dataframe(y)#.sample(100000, random_state=12345)
            try:
                year_start_index = time_idx("20{}-01-01 00:00:00".format(y))
            except IndexError:
                # If we're trying to get the index for before DATA_START, simply use 0
                year_start_index = 0
            year_end_index = time_idx("20{}-12-31 23:59:59".format(y))
            year_matrix = lil_matrix((MAX_POSSIBLE_STATIONS, year_end_index - year_start_index + 1), dtype=np.uint8)
            print("Processing trip data for year 20{}".format(y))
            for row in trips.itertuples():
                print("\rrow: {index:>11,}".format(index=row[0]), end="")
                station_id = row[STATION_ID_COL]
                start_time = row[START_TIME_COL].to_datetime()
                if station_id not in station_idx:
                    station_idx[station_id] = max_station_idx + 1
                    max_station_idx += 1
                row_station_idx = station_idx[station_id]
                # Remember to shift the index to this years numbering
                row_time_idx = time_idx(start_time) - year_start_index

                old_value = year_matrix[row_station_idx, row_time_idx]
                year_matrix[row_station_idx, row_time_idx] = old_value + 1
                # Our data type only goes to 255, so if we need more space, print it out
                if old_value > 250:
                    print("Getting close to an overflow!!! value:{}".format(old_value + 1))

            start_time_matrix = hstack([start_time_matrix, year_matrix], dtype=np.uint8)
            print("\nYear 20{} trip processing finished".format(y))
        actual_end_station_count = len(station_idx)
        actual_end_time_count = time_idx(trips['starttime'].max().to_datetime())
        final_matrix = (start_time_matrix.tolil()[0:actual_end_station_count+1, 0:actual_end_time_count+1]).tocsr()

        # Write newly built matrix to file system
        attributes = {  'station_idx': station_idx, 'data': final_matrix.data, 'indices':final_matrix.indices, 
                        'indptr':final_matrix.indptr, 'shape':final_matrix.shape }
        np.savez(start_time_matrix_npz, **attributes)

    # Return newly built start time matrix
    return final_matrix, station_idx, time_idx, time_at_idx

def load_stop_time_matrix():
    # We'll use the same station mapping and shape as the start_time_matrix
    start_time_matrix, station_idx, time_idx, _ = load_start_time_matrix()

    stop_time_matrix_npz = os.path.join(data_dir, "stop_time_matrix.npz")
    final_matrix = csr_matrix((0,0), dtype=np.uint8)

    if os.path.isfile(stop_time_matrix_npz):
        print("{} already exists. Skipping creation".format(stop_time_matrix_npz))
        with np.load(stop_time_matrix_npz) as loader:
            final_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    else:
        STATION_ID_COL = 4
        STOP_TIME_COL = 2
        stop_time_matrix = lil_matrix(start_time_matrix.shape, dtype=np.uint8)

        skipped_records = 0
        for y in YEARS:
            trips = load_trips_dataframe(y)#.sample(1000, random_state=12345)
            print("Processing trip data for year 20{}".format(y))
            for row in trips.itertuples():
                print("\rrow: {index:>11,}".format(index=row[0]), end="")
                station_id = row[STATION_ID_COL]
                stop_time = row[STOP_TIME_COL].to_datetime()
                if station_id not in station_idx:
                    skipped_records += 1
                    continue
                row_station_idx = station_idx[station_id]
                row_time_idx = time_idx(stop_time)

                # Ignore counts that end after the matrix ends (unfortunate but won't really affect our analysis much)
                if row_time_idx < stop_time_matrix.shape[1]:
                    old_value = stop_time_matrix[row_station_idx, row_time_idx]
                    stop_time_matrix[row_station_idx, row_time_idx] = old_value + 1
                    # Our data type only goes to 255, so if we need more space, print it out
                    if old_value > 250:
                        print("Getting close to an overflow!!! value:{}".format(old_value + 1))
                else:
                    skipped_records += 1
            print("\nYear 20{} trip processing finished".format(y))
        if skipped_records > 0:
            print("Skipped {} record(s)".format(skipped_records))
        final_matrix = stop_time_matrix.tocsr()

        # Write newly built matrix to file system
        attributes = { 'data': final_matrix.data, 'indices':final_matrix.indices, 
                       'indptr':final_matrix.indptr, 'shape':final_matrix.shape }
        np.savez(stop_time_matrix_npz, **attributes)

    # Return newly built stop time matrix
    return final_matrix, station_idx, time_idx


# Helper Methods
def month_indices(reduction_interval=1):
    # Return the month indices between July 2013 and September 2016
    return np.array([ time_idx("201{y}-{m:0>2}-01 00:00:00".format(y=y,m=m)) 
             for y in range(3,7) for m in range(1,13) 
             if (y > 3 or m > 6) and (y < 6 or m < 10)]) / reduction_interval
def week_indices(start_time_matrix, reduction_interval=1):
    total_buckets = start_time_matrix.shape[1]
    week_indices = np.arange(np.floor(total_buckets / INTERVAL_WEEKLY)) * INTERVAL_WEEKLY
    return week_indices / reduction_interval

def year_labels(reduction_interval=1,date_ticks=True):
    # Take the original month tickmarks and divide them by interval to find the appropriate
    # marks for the transformed data
    tickmarks = month_indices(reduction_interval)
    if date_ticks:
        tickmarks = [ time_at_idx(i * reduction_interval) for i in tickmarks ]
    ticklabels = ['']*len(tickmarks)
    ticklabels[0] = "2013"
    ticklabels[6] = "2014"
    ticklabels[18] = "2015"
    ticklabels[30] = "2016"
    return tickmarks, ticklabels

# Data Transformation utils
INTERVAL_DAILY = 48
INTERVAL_WEEKLY = INTERVAL_DAILY * 7
INTERVAL_YEARLY = INTERVAL_WEEKLY * 52
def get_total_weekly_trips(time_matrix):
    return get_agg_trips_by_interval(time_matrix, INTERVAL_WEEKLY)


def get_agg_trips_by_interval(time_matrix, interval=INTERVAL_DAILY, aggregator_fn=np.sum):
    # Aggregate along both the interval axis and the station axis
    # Leaving array over the years
    return _reshape_and_aggregate(time_matrix, interval, aggregator_fn)

def get_station_agg_trips_over_week(time_matrix, aggregator_fn=np.sum):
    # Aggregate along just the year axis
    # Leaving matrix of stations x week
    n_stations, total_buckets = time_matrix.shape
    interval = INTERVAL_WEEKLY
    end_index = math.floor(total_buckets / interval) * interval
    temp_trips = time_matrix[:, :end_index].todense().A

    first_data_idx = np.zeros((n_stations,1))
    for i in range(0,n_stations):
        non_zero_trips = temp_trips[i,:].nonzero()[0]
        if len(non_zero_trips) > 0:
            first_data_idx[i] = non_zero_trips[0]
        else:
            first_data_idx[i] = end_index - 1

    first_data_weekly_idx = np.floor(first_data_idx / interval)

    # Reshape the matrix so we have a 3rd dimension for the weekly data
    temp_trips = temp_trips.reshape((n_stations, -1, interval))

    # Aggregate along the first dim
    final_matrix = np.zeros((n_stations, interval))
    for i in range(0,n_stations):
        final_matrix[i,:] = aggregator_fn(temp_trips[i, int(first_data_weekly_idx[i]):, :], axis=0)

    return final_matrix

def construct_active_stations_by_bucket(start_time_matrix):
    # This works because we assume that the stations are ordered in start_time_matrix
    # by their 'activation' date. This assumption is currently correct because when we
    # build the start time matrix we go through the trip log, which is ordered in this manner
    active_stations = np.zeros(start_time_matrix.shape[1], dtype=np.int32)
    current_num = 0
    for i in range(active_stations.shape[0]):
        for j in range(current_num, start_time_matrix.shape[0]):
            if start_time_matrix[j,i] > 0:
                current_num += 1
            else:
                break
        active_stations[i] = current_num
    return active_stations



def get_agg_trips_over_day(time_matrix, aggregator_fn=np.sum):
    # This aggregates across the years into a station x 30 min throughout day matrix
    # then agreggates to a 30 min throughout day array
    return _reshape_and_aggregate(time_matrix, INTERVAL_DAILY, aggregator_fn, axes=[1,0])

def _reshape_and_aggregate(time_matrix, interval, aggregator_fn, axes=[2,0]):
    n_stations, total_buckets = time_matrix.shape
    if time_matrix.shape[1] % interval != 0:
        end_index = math.floor(total_buckets / interval) * interval
        temp_trips = time_matrix[:, :end_index].todense().A
    else:
        temp_trips = time_matrix.todense().A
    # Reshape the matrix so we have a 3rd dimension for the weekly data
    temp_trips = temp_trips.reshape((n_stations, -1, interval))
    # Aggregate along the first dim
    temp_trips = aggregator_fn(temp_trips, axis=axes[0])
    if len(axes) > 1:
        # Aggregate along the second dim
        temp_trips = aggregator_fn(temp_trips, axis=axes[1])
    return temp_trips

def normalize(time_matrix):
    # Normalize the stations by for each station, take the min or max
    # and divide the rest of the values by the absolute value of that
    maxes = np.max(np.abs(time_matrix), axis=1)
    maxes = np.repeat(maxes, time_matrix.shape[1]).reshape(time_matrix.shape)
    return np.divide(time_matrix, maxes)

def round(time_matrix, nearest_fraction=2):
    return np.round(time_matrix * nearest_fraction) / nearest_fraction
