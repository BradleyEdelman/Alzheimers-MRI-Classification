import tempfile, io, pickle, boto3, keras
from keras.models import load_model

def save_model_s3(model, bucket_name, s3_file_path, dbutils):
    """
    Save a Keras model to an S3 bucket.

    Parameters:
    model (keras.Model): The Keras model to save.
    bucket_name (str): The name of the S3 bucket.
    s3_file_path (str): The S3 file path where the model will be saved.
    dbutils (Databricks Utilities): Databricks utilities to access secrets.

    Returns:
    None
    """

    ACCESS_KEY = dbutils.secrets.get(scope="brad-aws", key="access_key")
    SECRET_KEY= dbutils.secrets.get(scope="brad-aws", key="secret_key")

    with tempfile.NamedTemporaryFile(suffix='.h5') as temp_file:
        model.save(temp_file.name)
        temp_file.seek(0)
        buffer = io.BytesIO(temp_file.read())

        s3=boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
        s3.upload_fileobj(buffer, bucket_name, s3_file_path)

    print(f"Model uploaded to S3 bucket: {bucket_name}{s3_file_path}")


def load_model_s3(bucket_name, s3_file_path, dbutils):
    """
    Load a Keras model from an S3 bucket.

    Parameters:
    bucket_name (str): The name of the S3 bucket.
    s3_file_path (str): The S3 file path where the model is saved.
    dbutils (Databricks Utilities): Databricks utilities to access secrets.

    Returns:
    keras.Model: The loaded Keras model.
    """

    ACCESS_KEY = dbutils.secrets.get(scope="brad-aws", key="access_key")
    SECRET_KEY = dbutils.secrets.get(scope="brad-aws", key="secret_key")

    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    with tempfile.NamedTemporaryFile(suffix='.h5') as temp_file:
        s3.download_fileobj(bucket_name, s3_file_path, temp_file)
        temp_file.seek(0)
        model = load_model(temp_file.name)
    
    return model


def save_pickle_s3(history, bucket_name, s3_file_path, dbutils):
    """
    Save a Python object to an S3 bucket in pickle format.

    Parameters:
    history (object): The Python object to save.
    bucket_name (str): The name of the S3 bucket.
    s3_file_path (str): The S3 file path where the object will be saved.
    dbutils (Databricks Utilities): Databricks utilities to access secrets.

    Returns:
    None
    """

    ACCESS_KEY = dbutils.secrets.get(scope="brad-aws", key="access_key")
    SECRET_KEY= dbutils.secrets.get(scope="brad-aws", key="secret_key")

    with tempfile.NamedTemporaryFile(suffix='.pkl') as temp_file:
        pickle.dump(history, temp_file)
        temp_file.seek(0)
        buffer = io.BytesIO(temp_file.read())

    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    s3.upload_fileobj(buffer, bucket_name, s3_file_path)

    print(f"Data uploaded to S3 bucket: {bucket_name}/{s3_file_path}")


def load_pickle_s3(bucket_name, s3_file_path, dbutils):
    """
    Load a Python object from an S3 bucket in pickle format.

    Parameters:
    bucket_name (str): The name of the S3 bucket.
    s3_file_path (str): The S3 file path where the object is saved.
    dbutils (Databricks Utilities): Databricks utilities to access secrets.

    Returns:
    object: The loaded Python object.
    """
    
    ACCESS_KEY = dbutils.secrets.get(scope="brad-aws", key="access_key")
    SECRET_KEY = dbutils.secrets.get(scope="brad-aws", key="secret_key")

    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    with tempfile.NamedTemporaryFile(suffix='.pkl') as temp_file:
        s3.download_fileobj(bucket_name, s3_file_path, temp_file)
        temp_file.seek(0)
        data = pickle.load(temp_file)

    return data