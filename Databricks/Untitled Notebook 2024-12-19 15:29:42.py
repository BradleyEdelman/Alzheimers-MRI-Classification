# Databricks notebook source
aws_keys_df = spark.read.format("csv").option("header", "true").option("sep", ",").load("/FileStore/tables/Brad_databricks_personal_accessKeys.csv")

ACCESS_KEY = aws_keys_df.collect()[0][0]
SECRET_KEY = aws_keys_df.collect()[0][1]

AWS_S3_BUCKET = "databricks-workspace-stack-a74b8-bucket"

MOUNT_NAME = "/mnt/AD_classification"

SOURCE_URL = f"s3a://{AWS_S3_BUCKET}"
EXTRA_CONFIGS = {
    "fs.s3a.access.key": ACCESS_KEY,
    "fs.s3a.secret.key": SECRET_KEY
}

dbutils.fs.mount(SOURCE_URL, MOUNT_NAME, extra_configs=EXTRA_CONFIGS)

# COMMAND ----------

import pandas as pd
import numpy as np
import cv2
train = pd.read_parquet("/dbfs/mnt/AD_classification/train-00000-of-00001-c08a401c53fe5312.parquet")


def dict_to_image(image_dict):
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        byte_string = image_dict['bytes']
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")

train['img_arr'] = train['image'].apply(dict_to_image)
train.drop("image", axis=1, inplace=True)
train.head()

# COMMAND ----------

dbutils.fs.unmount(MOUNT_NAME)
