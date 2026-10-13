# Sumber data

1. Data smart meter: London Datastore, "SmartMeter Energy Consumption Data in London Households"
   (UK Power Networks, lisensi CC BY). Berkas yang dipakai: `low-carbon-london-data-168-files`,
   file pertama `LCL-June2015v2_0.csv` (1 juta baris, 30 rumah tangga).
   https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d
   Berkas CSV tidak disimpan di repositori karena ukurannya (lihat .gitignore).
2. Cuaca: Open-Meteo Historical Weather API, titik London 51.5N 0.12W, per jam, 1 Nov 2011 - 28 Feb 2014.
   Berkas hasil unduhan: `weather_london_2013.json`.
3. Hari libur Inggris: `uk_bank_holidays.csv`, disusun sendiri, belum diverifikasi penuh terhadap sumber resmi.

Unduh ulang semua data dengan: `bash tools/download_data.sh`
