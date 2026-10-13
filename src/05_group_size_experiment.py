"""Perluasan: pengaruh ukuran kelompok rumah (k) terhadap akurasi peramalan.
Menguji pernyataan artikel bahwa peramalan membaik bila jumlah rumah bertambah.
Untuk tiap k, ambil k rumah dengan rekaman valid terbanyak (deterministik), agregasi, lalu latih Ridge dan DTR depth 5."""
import argparse, json
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, PolynomialExpansion, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from common import get_spark, SEED

p = argparse.ArgumentParser()
p.add_argument("--blocks", default="data/raw/halfhourly/LCL-June2015v2_0.csv")
p.add_argument("--ks", default="5,10,15,20,30")
p.add_argument("--ridge-reg", type=float, default=1e-3)
a = p.parse_args()
spark = get_spark("groupsize")
prep = spark.read.parquet("data/processed/agg.parquet")  # hanya untuk memastikan data olahan ada
raw = spark.read.option("header", True).csv(a.blocks)
tc = [c for c in raw.columns if c.lower() in ("datetime", "tstp")][0]
ec = [c for c in raw.columns if c.lower().startswith(("energy", "kwh"))][0]
df = (raw.select("LCLid", F.to_timestamp(F.substring(F.col(f"`{tc}`"), 1, 19)).alias("tstp"),
                 F.col(f"`{ec}`").cast("double").alias("energy"))
      .dropna().filter(F.col("energy") > 0))
import pandas as pd, json as _j
w = pd.DataFrame(_j.load(open("data/raw/weather_london_2013.json"))["hourly"])
w["hour_ts"] = pd.to_datetime(w["time"]); w = w[["hour_ts", "apparent_temperature"]]
wd = spark.createDataFrame(w)
hol = pd.read_csv("data/raw/uk_bank_holidays.csv"); hol.columns = ["date", "type"]
hol_dates = sorted(set(pd.to_datetime(hol["date"]).dt.strftime("%Y-%m-%d")))
rank = df.groupBy("LCLid").count().orderBy(F.desc("count"), "LCLid").collect()
ids = [r["LCLid"] for r in rank]

ohe = OneHotEncoder(inputCols=["hour"], outputCols=["hour_ohe"], dropLast=False)
asm = VectorAssembler(inputCols=["n_houses", "apparent_temperature", "hour_ohe", "weekend"], outputCol="raw")
poly = PolynomialExpansion(degree=2, inputCol="raw", outputCol="poly")
scl = StandardScaler(inputCol="poly", outputCol="features", withMean=False, withStd=True)
asm_t = VectorAssembler(inputCols=["n_houses", "apparent_temperature", "hour", "weekend"], outputCol="features")
ev = {m: RegressionEvaluator(labelCol="sum_energy", metricName=m) for m in ["r2", "mae", "rmse"]}
rows = []
for k in [int(x) for x in a.ks.split(",")]:
    sel = df.filter(F.col("LCLid").isin(ids[:k]))
    g = (sel.groupBy("tstp").agg(F.countDistinct("LCLid").alias("n_houses"), F.sum("energy").alias("sum_energy"))
         .withColumn("hour", F.hour("tstp")).withColumn("hdate", F.date_format("tstp", "yyyy-MM-dd"))
         .withColumn("weekend", F.when(F.dayofweek("tstp").isin(1, 7) | F.col("hdate").isin(hol_dates), 1.0).otherwise(0.0))
         .withColumn("hour_ts", F.date_trunc("hour", "tstp")).join(wd, "hour_ts").dropna()
         .withColumn("n_houses", F.col("n_houses").cast("double")))
    tr, te = g.randomSplit([0.8, 0.2], seed=SEED); tr.cache(); te.cache()
    mean_y = te.agg(F.avg("sum_energy")).first()[0]
    for name, stages in [("Ridge", [ohe, asm, poly, scl, LinearRegression(labelCol="sum_energy", regParam=a.ridge_reg, elasticNetParam=0.0, maxIter=200)]),
                         ("DTR_depth5", [asm_t, DecisionTreeRegressor(labelCol="sum_energy", maxDepth=5, seed=SEED)])]:
        pr = Pipeline(stages=stages).fit(tr).transform(te)
        m = {n: e.evaluate(pr) for n, e in ev.items()}
        rows.append({"k_rumah": k, "model": name, "r2": round(m["r2"], 4), "mae": round(m["mae"], 4),
                     "rmse": round(m["rmse"], 4), "rmse_relatif_pct": round(100 * m["rmse"] / mean_y, 1),
                     "mean_sum_energy": round(mean_y, 3), "n_test": te.count()})
        print(rows[-1])
    tr.unpersist(); te.unpersist()
json.dump(rows, open("results/group_size_experiment.json", "w"), indent=2)
