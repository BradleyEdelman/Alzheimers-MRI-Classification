import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def global_prune_model(model, pruning_factor=0.2):
    """
    Prune weights of all layers (including ResNet50 sub-model) by removing weights 
    below the pruning threshold, calculated based on the lowest X% of weights.
    
    Parameters:
    - model: The TensorFlow model with layers to prune (including ResNet50).
    - pruning_factor: The percentage of weights to prune, e.g., 0.2 for 20%.
    """
    
    if pruning_factor != 0:

        # Collect weights from model layers and sub-layers
        all_weights = []
        layers = []
        
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):  # If there is a sub-model/backbone (e.g. ResNet50)
                for sub_layer in layer.layers:  # Access sub-layers
                    if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                        layers.append(sub_layer)
                        all_weights.append(sub_layer.get_weights()[0].flatten())  # Only weights (not biases)
            else:
                layers.append(layer)
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                    all_weights.append(layer.get_weights()[0].flatten())
        
        # Concatenate all weights and compute the global threshold
        all_weights_flat = np.concatenate(all_weights)
        sorted_weights = np.sort(np.abs(all_weights_flat))
        threshold = sorted_weights[int(len(sorted_weights) * pruning_factor)]
        
        # Prune weights lower than the threshold 
        weight_pointer = 0
        for layer in layers:
            if isinstance(layer, tf.keras.Model):  # Sub-model
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                        
                        weights, biases = sub_layer.get_weights()
                        weight_flat = weights.flatten()
                        weight_mask = np.abs(weight_flat) > threshold  # Mask for pruning
                        pruned_weights = weight_flat * weight_mask
                        pruned_weights = pruned_weights.reshape(weights.shape)  # Reshape back to original shape
                        sub_layer.set_weights([pruned_weights, biases])  # Set pruned weights
                        weight_pointer += 1
            else:
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):

                    weights, biases = layer.get_weights()
                    weight_flat = weights.flatten()
                    weight_mask = np.abs(weight_flat) > threshold
                    pruned_weights = weight_flat * weight_mask
                    pruned_weights = pruned_weights.reshape(weights.shape)
                    layer.set_weights([pruned_weights, biases])
                    weight_pointer += 1

        print(f"Pruning completed. The bottom {pruning_factor * 100}% of weights have been removed.")

    return model


def plot_weight_distribution(pruned_model, title="Weight Distribution"):
    """
    Plot the distribution of weights in the pruned model.
    
    Parameters:
    - pruned_model: The TensorFlow model after pruning.
    - title: The title of the plot.
    """
    
    all_weights = []
    for layer in pruned_model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights, _ = layer.get_weights()
            all_weights.append(weights.flatten())  # Flatten the weights to make them 1D

    all_weights_flat = np.concatenate(all_weights)  # Combine all weights into a single array
    sns.histplot(all_weights_flat, kde=True, bins=50)  # Plot histogram and KDE
    plt.title(title)
    plt.xlabel('Weight values')
    plt.ylabel('Frequency')
    plt.show()