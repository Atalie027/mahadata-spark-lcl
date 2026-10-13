"""Tahap 1-2 artikel: pengumpulan, pembersihan, dan agregasi data.

Menggabungkan data smart meter, cuaca (Open-Meteo), dan hari libur, membuang nilai null dan nol,
lalu mengagregasi per timestamp:
  n_houses   = Count(LCLid)  (jumlah rumah yang tercatat pada timestamp itu)
  sum_energy = Sum(energy)   (variabel target)
"""
import argparse, glob, json, os
import pandas as pd
from pyspark.sql import functions as F
from common import get_spark

p = argparse.ArgumentParser()
p.add_argument("--blocks", default="data/raw/halfhourly/LCL-June2015v2_0.csv")
p.add_argument("--weather", default="data/raw/weather_london_2013.json")
p.add_argument("--holidays", default="data/raw/uk_bank_holidays.csv")
p.add_argument("--out", default="data/processed/agg.parquet")
p.add_argument("--n-houses", type=int, default=0,
               help="jumlah rumah dalam kelompok (0 = semua rumah di file)")
a = p.parse_args()

spark = get_spark("prepare")
files = sorted(glob.glob(a.blocks))
assert files, f"Tidak ada file untuk pola {a.blocks}"
print("File data:", files)

raw = spark.read.option("header", True).csv(files)
time_col = [c for c in raw.columns if c.lower() in ("datetime", "tstp")][0]
energy_col = [c for c in raw.columns if c.lower().startswith(("energy", "kwh"))][0]
df = (raw.select(
        F.col("LCLid"),
        F.to_timestamp(F.substring(F.col(f"`{time_col}`"), 1, 19)).alias("tstp"),
        F.col(f"`{energy_col}`").cast("double").alias("energy"))
      .dropna(subset=["tstp", "energy"])
      .filter(F.col("energy") > 0))  # nol = pemadaman/gangguan komunikasi (sesuai artikel)

if a.n_houses > 0:
    top = df.groupBy("LCLid").count().orderBy(F.desc("count"), "LCLid").limit(a.n_houses)
    df = df.join(top.select("LCLid"), "LCLid")
df.createOrReplaceTempView("meter")

# --- cuaca Open-Meteo (pengganti Darksky) ---
w = json.load(open(a.weather))["hourly"]
wdf = pd.DataFrame(w).rename(columns={"time": "hour_ts", "temperature_2m": "temperature",
        "apparent_temperature": "apparent_temperature", "dew_point_2m": "dew_point",
        "relative_humidity_2m": "humidity", "wind_speed_10m": "wind_speed",
        "surface_pressure": "pressure"})
wdf["hour_ts"] = pd.to_datetime(wdf["hour_ts"])
spark.createDataFrame(wdf).createOrReplaceTempView("weather")

# --- hari libur ---
h = pd.read_csv(a.holidays)
h.columns = ["date", "type"][: len(h.columns)]
h["date"] = pd.to_datetime(h["date"]).dt.strftime("%Y-%m-%d")
spark.createDataFrame(h[["date"]].drop_duplicates()).createOrReplaceTempView("holidays")

# --- agregasi dengan Spark SQL ---
agg = spark.sql("""
SELECT m.tstp,
       COUNT(DISTINCT m.LCLid)            AS n_houses,
       SUM(m.energy)                      AS sum_energy,
       HOUR(m.tstp)                       AS hour,
       MONTH(m.tstp)                      AS month,
       CASE WHEN DAYOFWEEK(m.tstp) IN (1,7) OR h.date IS NOT NULL THEN 1 ELSE 0 END AS weekend,
       FIRST(w.temperature)               AS temperature,
       FIRST(w.apparent_temperature)      AS apparent_temperature,
       FIRST(w.dew_point)                 AS dew_point,
       FIRST(w.humidity)                  AS humidity,
       FIRST(w.wind_speed)                AS wind_speed,
       FIRST(w.pressure)                  AS pressure
FROM meter m
LEFT JOIN weather w  ON w.hour_ts = DATE_TRUNC('hour', m.tstp)
LEFT JOIN holidays h ON h.date = DATE_FORMAT(m.tstp, 'yyyy-MM-dd')
GROUP BY m.tstp, h.date
""").dropna(subset=["apparent_temperature"])

agg.write.mode("overwrite").parquet(a.out)
n_raw, n_clean, n_agg = raw.count(), df.count(), spark.read.parquet(a.out).count()
stats = {"raw_rows": n_raw, "clean_rows": n_clean, "agg_rows": n_agg,
         "removed_pct": round(100 * (1 - n_clean / n_raw), 3), "files": files}
os.makedirs("results", exist_ok=True)
json.dump(stats, open("results/prepare_stats.json", "w"), indent=2)
print(stats)
