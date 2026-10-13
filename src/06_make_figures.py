"""Membuat grafik hasil untuk laporan dari berkas di results/."""
import json, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
F = "figures"
# 1) aktual vs prediksi (Ridge, DTR5) pada 6 hari di set uji (analog Gambar 14 artikel)
r = pd.read_csv("results/pred_Ridge.csv", parse_dates=["tstp"]); d = pd.read_csv("results/pred_DTR_depth5.csv", parse_dates=["tstp"])
lo, hi = pd.Timestamp("2013-03-14"), pd.Timestamp("2013-03-20")
r, d = r[(r.tstp >= lo) & (r.tstp < hi)], d[(d.tstp >= lo) & (d.tstp < hi)]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(r.tstp, r.sum_energy, ".", ms=3, color="#888", label="Aktual (titik uji)")
ax.plot(r.tstp, r.prediction, ".", ms=3, color="#4C78A8", label="Ridge")
ax.plot(d.tstp, d.prediction, ".", ms=3, color="#E45756", label="DTR depth 5")
ax.set(ylabel="Sum energy (kWh/30 mnt)", title="Aktual vs prediksi, 14-19 Maret 2013 (data uji saja)"); ax.legend(); fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(f"{F}/fig_actual_vs_pred.png", dpi=150); plt.close(fig)
# 2) ukuran kelompok
g = pd.DataFrame(json.load(open("results/group_size_experiment.json")))
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for m, c in [("Ridge", "#4C78A8"), ("DTR_depth5", "#E45756")]:
    s = g[g.model == m]; ax.plot(s.k_rumah, s.r2, marker="o", color=c, label=m)
ax.set(xlabel="Jumlah rumah dalam kelompok", ylabel="R² (data uji)", title="Pengaruh ukuran kelompok terhadap R²"); ax.legend()
fig.tight_layout(); fig.savefig(f"{F}/fig_group_size.png", dpi=150); plt.close(fig)
# 3) partition
p = pd.DataFrame(json.load(open("results/partition_experiment.json")))
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for m, c in [("Ridge", "#4C78A8"), ("DTR_depth5", "#E45756")]:
    s = p[p.model == m]; ax.plot(s.partitions, s.fit_median_s, marker="o", color=c, label=m)
ax.set(xscale="log", xlabel="Jumlah partition", ylabel="Waktu fitting, median (detik)", title="Waktu fitting vs jumlah partition"); ax.legend()
fig.tight_layout(); fig.savefig(f"{F}/fig_partitions.png", dpi=150); plt.close(fig)
# 4) waktu fitting per model
m = json.load(open("results/model_results.json"))["models"]
fig, ax = plt.subplots(figsize=(6, 3.4)); ax.bar(list(m), [v["fit_seconds"] for v in m.values()], color="#4C78A8")
ax.set(ylabel="Waktu fitting + tuning (detik)", title="Waktu fitting per model"); plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
fig.tight_layout(); fig.savefig(f"{F}/fig_fit_times.png", dpi=150); plt.close(fig)
print("grafik dibuat")
