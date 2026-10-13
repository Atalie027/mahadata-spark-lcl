#!/usr/bin/env bash
# Mengunduh data cuaca dan data smart meter (file pertama dari 168 file) ke data/raw/
set -e
mkdir -p data/raw/halfhourly
curl -s -o data/raw/weather_london_2013.json "https://archive-api.open-meteo.com/v1/archive?latitude=51.5&longitude=-0.12&start_date=2011-11-01&end_date=2014-02-28&hourly=temperature_2m,apparent_temperature,dew_point_2m,relative_humidity_2m,wind_speed_10m,surface_pressure&timezone=Europe%2FLondon"
curl -L -sS -o /tmp/lcl168.zip "https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/Partitioned%20LCL%20Data.zip"
unzip -j -o /tmp/lcl168.zip "*LCL-June2015v2_0.csv" -d data/raw/halfhourly
rm /tmp/lcl168.zip
