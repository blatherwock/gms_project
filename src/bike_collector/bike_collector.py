from __future__ import print_function

from datetime import datetime
from urllib2 import urlopen
from cStringIO import StringIO
import json
import boto3

SITE = 'https://www.citibikenyc.com/stations/json'  # URL of the site to check
s3 = boto3.resource("s3")
bucket = 'nyc-citibike-station-data'
key_prefix = 'raw/'

columns = {'id':'station_id',
           'lastCommunicationTime':'datatime',
           'availableBikes':'avail_bikes',
           'availableDocks':'avail_docks',
           'totalDocks':'tot_docks',
           'statusKey':'status_key'}
column_order = ['station_id', 'datatime', 'avail_bikes', 'avail_docks',
                'tot_docks', 'status_key']

def lambda_handler(event, context):
    print('Checking {} at {}...'.format(SITE, event['time']))
    try:
        # Read data from remote endpoint
        data = json.load(urlopen(SITE))
        cleaned_data = [{columns[k]:v for k,v in obj.iteritems() 
                        if k in columns.keys()} 
                        for obj in data['stationBeanList']]

        header = ', '.join(column_order) + '\n'
        data_rows = [','.join([str(row[col]) for col in column_order]) for row in cleaned_data]
        body = header + '\n'.join(data_rows)

        # Save data to S3
        timestamp = data['executionTime'].replace(" ", '_')
        key = key_prefix + timestamp + '.csv'
        handle = StringIO(body)
        s3.Bucket(bucket).put_object(Key=key, Body=handle.read())
    except:
        print('Loading data failed!')
        raise
    
if __name__ == '__main__':
    lambda_handler({'time':'now'}, None)