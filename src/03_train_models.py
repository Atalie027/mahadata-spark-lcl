"""Tahap 5 artikel: model Spark MLlib.

Model: regresi polinomial multivariat (plain, Lasso, Ridge, Elastic Net) dan decision tree regressor.
Hiperparameter di-tuning dengan CrossValidator (metrik R2, seperti artikel).
Fitur: n_houses, apparent_temperature, hour (one-hot), weekend. Target: sum_energy.
Asumsi (tidak dinyatakan artikel, ditetapkan sendiri): split 80/20 acak seed 42, 3-fold CV,
polinomial derajat 2, StandardScaler setelah ekspansi.
"""
import argparse, json, time
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, PolynomialExpansion, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from common import get_spark, SEED

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/processed/agg.parquet")
p.add_argument("--out", default="results/model_results.json")
p.add_argument("--degree", type=int, default=2)
p.add_argument("--folds", type=int, default=3)
p.add_argument("--dtr-depths", default="5,12")
p.add_argument("--partitions", type=int, default=0, help="0 = biarkan default")
a = p.parse_args()

spark = get_spark("train")
df = spark.read.parquet(a.input).select("tstp", "n_houses", "apparent_temperature", "hour", "weekend", "sum_energy")
df = df.withColumn("n_houses", F.col("n_houses").cast("double")) \
       .withColumn("weekend", F.col("weekend").cast("double"))
if a.partitions:
    df = df.repartition(a.partitions)
train, test = df.randomSplit([0.8, 0.2], seed=SEED)
train.cache(); n_tr, n_te = train.count(), test.count()
print(f"train={n_tr} test={n_te}")

ohe = OneHotEncoder(inputCols=["hour"], outputCols=["hour_ohe"], dropLast=False)
asm = VectorAssembler(inputCols=["n_houses", "apparent_temperature", "hour_ohe", "weekend"], outputCol="raw")
poly = PolynomialExpansion(degree=a.degree, inputCol="raw", outputCol="poly")
scl = StandardScaler(inputCol="poly", outputCol="features", withMean=False, withStd=True)
asm_tree = VectorAssembler(inputCols=["n_houses", "apparent_temperature", "hour", "weekend"], outputCol="features")
ev = {m: RegressionEvaluator(labelCol="sum_energy", predictionCol="prediction", metricName=m)
      for m in ["r2", "mae", "mse", "rmse"]}

def lr(**kw): return LinearRegression(featuresCol="features", labelCol="sum_energy", maxIter=200, **kw)
regs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
specs = {  # nama -> (estimator, pipeline-stages-awal, grid)
    "Plain": (lr(regParam=0.0), [ohe, asm, poly, scl], None),
    "Lasso": (lr(elasticNetParam=1.0), [ohe, asm, poly, scl], lambda e: ParamGridBuilder().addGrid(e.regParam, regs).build()),
    "Ridge": (lr(elasticNetParam=0.0), [ohe, asm, poly, scl], lambda e: ParamGridBuilder().addGrid(e.regParam, regs).build()),
    "ElasticNet": (lr(), [ohe, asm, poly, scl], lambda e: ParamGridBuilder().addGrid(e.regParam, regs)
                                                       .addGrid(e.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0]).build()),
}
for d in [int(x) for x in a.dtr_depths.split(",")]:
    # depth 5 = konfigurasi artikel; depth lain = perluasan (artikel menyebut depth 12 untuk semua rumah)
    specs[f"DTR_depth{d}"] = (DecisionTreeRegressor(featuresCol="features", labelCol="sum_energy",
                                                    maxDepth=d, seed=SEED), [asm_tree], None)

results = {"n_train": n_tr, "n_test": n_te, "degree": a.degree, "folds": a.folds, "models": {}}
for name, (est, stages, gridf) in specs.items():
    pipe_stages = stages + [est]
    t0 = time.time()
    if gridf:
        cv = CrossValidator(estimator=Pipeline(stages=pipe_stages), estimatorParamMaps=gridf(est),
                            evaluator=ev["r2"], numFolds=a.folds, seed=SEED, parallelism=1)
        model = cv.fit(train)
        best = model.bestModel
        cv_r2 = max(model.avgMetrics)
    else:
        best = Pipeline(stages=pipe_stages).fit(train); cv_r2 = None
    fit_s = time.time() - t0
    t1 = time.time(); pred = best.transform(test); pred.cache(); pred.count(); test_s = time.time() - t1
    m = {k: v.evaluate(pred) for k, v in ev.items()}
    last = best.stages[-1]
    tuned = {}
    if name != "Plain" and not name.startswith("DTR"):
        tuned = {"regParam": last.getRegParam(), "elasticNetParam": last.getElasticNetParam()}
    results["models"][name] = {**m, "cv_r2": cv_r2, "fit_seconds": round(fit_s, 2),
                               "test_seconds": round(test_s, 3), "tuned": tuned}
    print(name, {k: round(v, 4) for k, v in m.items()}, tuned, f"fit={fit_s:.1f}s")
    if name in ("Ridge", "DTR_depth5"):  # simpan prediksi untuk grafik aktual vs prediksi
        pred.select("tstp", "n_houses", "prediction", "sum_energy").orderBy("tstp") \
            .toPandas().to_csv(f"results/pred_{name}.csv", index=False)
json.dump(results, open(a.out, "w"), indent=2)
