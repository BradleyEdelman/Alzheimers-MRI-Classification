# Databricks notebook source
def mount_aws_s3_bucket(AWS_S3_BUCKET, KEY_FILE):

    from pyspark.sql import SparkSession

    # Initialize the Spark session (if not already available)
    spark = SparkSession.builder.appName("tmp").getOrCreate()

    # extract aws credentials from hidden table 
    aws_keys_df = spark.read.format("csv").option("header", "true").option("sep", ",").load(KEY_FILE)

    ACCESS_KEY = aws_keys_df.collect()[0][0]
    SECRET_KEY = aws_keys_df.collect()[0][1]

    # specify bucket and mount point
    MOUNT_NAME = f"/mnt/{AWS_S3_BUCKET.split('/')[-2]}"
    SOURCE_URL = f"s3a://{AWS_S3_BUCKET}"
    EXTRA_CONFIGS = { "fs.s3a.access.key": ACCESS_KEY, "fs.s3a.secret.key": SECRET_KEY}

    # mount bucket
    if any(mount.mountPoint == MOUNT_NAME for mount in dbutils.fs.mounts()):
        print(f"{MOUNT_NAME} is already mounted.")
        return
    else:
        dbutils.fs.mount(SOURCE_URL, MOUNT_NAME, extra_configs = EXTRA_CONFIGS)
        print(f"{MOUNT_NAME} is now mounted.")

    return MOUNT_NAME

