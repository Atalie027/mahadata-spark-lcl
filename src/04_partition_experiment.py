"""Perluasan (panduan 4.1): pengaruh jumlah partition terhadap waktu fitting & testing.
Model Ridge (regParam tetap) dan DTR depth 5; tiap konfigurasi diulang dan diambil median."""
import argparse, json, statistics, time
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, PolynomialExpansion, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from common import get_spark, SEED

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/processed/agg.parquet")
p.add_argument("--parts", default="1,2,4,8,16,32")
p.add_argument("--reps", type=int, default=3)
p.add_argument("--ridge-reg", type=float, default=1e-3)
a = p.parse_args()
spark = get_spark("partitions")
base = spark.read.parquet(a.input).select("n_houses", "apparent_temperature", "hour", "weekend", "sum_energy") \
    .withColumn("n_houses", F.col("n_houses").cast("double")).withColumn("weekend", F.col("weekend").cast("double"))
train0, test0 = base.randomSplit([0.8, 0.2], seed=SEED)
train0.write.mode("overwrite").parquet("data/processed/_tr"); test0.write.mode("overwrite").parquet("data/processed/_te")
tr_src, te_src = spark.read.parquet("data/processed/_tr"), spark.read.parquet("data/processed/_te")

ohe = OneHotEncoder(inputCols=["hour"], outputCols=["hour_ohe"], dropLast=False)
asm = VectorAssembler(inputCols=["n_houses", "apparent_temperature", "hour_ohe", "weekend"], outputCol="raw")
poly = PolynomialExpansion(degree=2, inputCol="raw", outputCol="poly")
scl = StandardScaler(inputCol="poly", outputCol="features", withStd=True, withMean=False)
asm_t = VectorAssembler(inputCols=["n_houses", "apparent_temperature", "hour", "weekend"], outputCol="features")
pipes = {
    "Ridge": lambda: Pipeline(stages=[ohe, asm, poly, scl, LinearRegression(labelCol="sum_energy", regParam=a.ridge_reg, elasticNetParam=0.0, maxIter=200)]),
    "DTR_depth5": lambda: Pipeline(stages=[asm_t, DecisionTreeRegressor(labelCol="sum_energy", maxDepth=5, seed=SEED)]),
}
r2 = RegressionEvaluator(labelCol="sum_energy", metricName="r2")
rows = []
for n in [int(x) for x in a.parts.split(",")]:
    tr, te = tr_src.repartition(n).cache(), te_src.repartition(n).cache(); tr.count(); te.count()
    for name, mk in pipes.items():
        fits, tests, score = [], [], None
        for _ in range(a.reps):
            t0 = time.time(); m = mk().fit(tr); fits.append(time.time() - t0)
            t1 = time.time(); pr = m.transform(te); score = r2.evaluate(pr); tests.append(time.time() - t1)
        rows.append({"partitions": n, "model": name, "fit_median_s": round(statistics.median(fits), 3),
                     "test_median_s": round(statistics.median(tests), 3), "r2": round(score, 5)})
        print(rows[-1])
    tr.unpersist(); te.unpersist()
json.dump(rows, open("results/partition_experiment.json", "w"), indent=2)
