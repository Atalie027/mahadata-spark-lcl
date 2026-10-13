"""Tahap 3 artikel: analisis deskriptif dengan Spark SQL + seleksi fitur (korelasi Pearson)."""
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from common import get_spark

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/processed/agg.parquet")
p.add_argument("--fig", default="figures")
a = p.parse_args()
os.makedirs(a.fig, exist_ok=True)
spark = get_spark("descriptive")
spark.read.parquet(a.input).createOrReplaceTempView("agg")

# rata-rata energi per rumah (sum_energy / n_houses) agar sebanding antar timestamp
hourly = spark.sql("SELECT hour, AVG(sum_energy/n_houses) avg_kwh FROM agg GROUP BY hour ORDER BY hour").toPandas()
monthly = spark.sql("""SELECT month, AVG(sum_energy/n_houses) avg_kwh, AVG(apparent_temperature) avg_temp
                       FROM agg GROUP BY month ORDER BY month""").toPandas()
wk = spark.sql("SELECT weekend, AVG(sum_energy/n_houses) avg_kwh FROM agg GROUP BY weekend").toPandas()
hourly.to_csv("results/desc_hourly.csv", index=False)
monthly.to_csv("results/desc_monthly.csv", index=False)
wk.to_csv("results/desc_weekend.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 3.5)); ax.plot(hourly.hour, hourly.avg_kwh, marker="o")
ax.set(xlabel="Jam", ylabel="Rata-rata energi per rumah (kWh/30 mnt)", title="Profil konsumsi per jam")
fig.tight_layout(); fig.savefig(f"{a.fig}/fig_hourly.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.5)); ax.bar(monthly.month, monthly.avg_kwh, color="#4C78A8")
ax2 = ax.twinx(); ax2.plot(monthly.month, monthly.avg_temp, color="#E45756", marker="o")
ax.set(xlabel="Bulan", ylabel="Rata-rata energi per rumah"); ax2.set_ylabel("Suhu semu rata-rata (°C)")
fig.tight_layout(); fig.savefig(f"{a.fig}/fig_monthly.png", dpi=150); plt.close(fig)

# korelasi Pearson: cuaca vs sum_energy (Gambar 3 dan 4 artikel)
cols = ["temperature", "apparent_temperature", "dew_point", "humidity", "wind_speed", "pressure"]
pdf = spark.sql(f"SELECT sum_energy, n_houses, {','.join(cols)} FROM agg").toPandas()
cw = pdf[cols].corr().abs(); ce = pdf[cols + ["sum_energy"]].corr().abs()[["sum_energy"]].drop("sum_energy")
cw.to_csv("results/corr_weather.csv"); ce.to_csv("results/corr_energy.csv")
fig, ax = plt.subplots(figsize=(5.5, 4.5)); sns.heatmap(cw, annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Korelasi absolut antar variabel cuaca"); fig.tight_layout()
fig.savefig(f"{a.fig}/fig_corr_weather.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(3.5, 4)); sns.heatmap(ce, annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Korelasi dengan sum_energy"); fig.tight_layout()
fig.savefig(f"{a.fig}/fig_corr_energy.png", dpi=150); plt.close(fig)
print(monthly.round(3).to_string(index=False)); print(ce.round(3))
