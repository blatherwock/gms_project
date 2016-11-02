README for bike_collector.py
============================

bike_collector.py is a script that can be used in a AWS lambda function to query the citibike station status and save the result in S3.

To use
 1. Set the bucket and key variables to the bucket you have created for holding the raw data
 2. Create a lambda function that uses Python and copy the code from bike_collector.py to the code of the lambda function.
 3. Set a trigger for the function as a CloudWatch Event - Schedule for the rate that you desire.
 4. Test the function and confirm the data is being saved in your S3 bucket
 5. Enable the trigger to start the function running


Notes
-----
- I used a AWS Lambda template for setting up the initial function, it was a site canary test template.
- Note that the saved data removes redundant information about the stations like their names, latitude/longitude, and the text version of their statuses. Since the names and latitude/longitude of the stations don't change, these can be readded by inspecting the station id. Similarly the text status of 'In Service' can be readded by inspecting the service state key.
- The data is saved in a comma separated format and each query to the stations is saved in a new file.
- A different script must be used to aggregate these files together.
