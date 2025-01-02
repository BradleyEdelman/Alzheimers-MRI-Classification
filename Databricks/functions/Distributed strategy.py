# Databricks notebook source
def distributed_strategy():

    import tensorflow as tf
    import psutil

    # Get the number of logical CPU cores
    logical_cores = psutil.cpu_count(logical=True)
    print(f"Number of logical CPU cores: {logical_cores}")

    # List available GPU devices 
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"Available GPU devices: {gpu_devices}")

    # Create a list of devices
    devices = [f"/cpu:{i}" for i in range(logical_cores)]  # Use one device per logical core for CPUs
    devices.extend([f"/gpu:{i}" for i in range(len(gpu_devices))])  # Add all available GPUs

    print(f"All devices (CPUs and GPUs): {devices}")

    # compile a MirroredStrategy
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    print(f"Strategy initialized with devices: {strategy.num_replicas_in_sync} replicas")

    return strategy

