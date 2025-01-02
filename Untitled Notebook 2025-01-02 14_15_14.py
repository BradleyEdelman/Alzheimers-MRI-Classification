# Databricks notebook source
def augment_mri_image(images, training=True):
    
    import tensorflow as tf
    if training:
        images = tf.keras.layers.RandomFlip("horizontal_and_vertical")(images)
        images = tf.keras.layers.RandomRotation(0.2)(images)
        images = tf.keras.layers.RandomZoom(0.1)(images)
        images = tf.keras.layers.RandomTranslation(0.1, 0.1, 0.1)(images)
        images = images + tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=0.1, dtype=images.dtype)
    return images

# Example usage
# Assuming `mri_image` is a 3D NumPy array representing the MRI image with shape (depth, height, width)
# Normalize and expand dimensions to add the channel dimension (required for TensorFlow/Keras)
mri_image = tf.random.uniform((64, 64, 64), dtype=tf.float32)  # Dummy MRI image data for demonstration
mri_image = tf.expand_dims(mri_image, axis=-1)  # Add the channel dimension

# Apply data augmentation
augmented_image = augment_mri_image(mri_image, training=True)

# Convert back to NumPy array for further processing if needed
augmented_image_np = augmented_image.numpy()

print(f"Original shape: {mri_image.shape}")
print(f"Augmented shape: {augmented_image_np.shape}")
