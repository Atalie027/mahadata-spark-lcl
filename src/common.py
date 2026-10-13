"""Fungsi bersama: membuat SparkSession dan membaca konfigurasi."""
from pyspark.sql import SparkSession

SEED = 42  # random seed tetap untuk pembagian data dan cross-validation


def get_spark(app="lcl", shuffle_partitions=8, driver_mem="4g"):
    spark = (
        SparkSession.builder.master("local[*]").appName(app)
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # data mentah memuat string "Null"; mode ANSI dimatikan agar CAST menghasilkan NULL
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
