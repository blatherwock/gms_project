# Predicting CikiBike future locations
[CitiBike](https://www.citibikenyc.com/) is a bicycle sharing service in New
York City. Although they provide
[real-time statuses](https://feeds.citibikenyc.com/stations/stations.json) for
the number of bicycles at each station, a frustrating issue is knowing whether
there will be bikes or docks available at a station in the future.

This project provides a way of learning this information based on historical
data.

## Setup
Dependencies:
  - Python 3

After installing dependencies:
  - ``git clone https://github.com/pedropobla/gms_project.git``
  - ``cd gms_project``
  - ``pyvenv venv``
  - ``source venv/bin/activate``
  - ``pip3 install -r requirements.txt``
  - ``make``