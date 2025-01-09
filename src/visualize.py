import matplotlib.pyplot as plt

def visualize_training(history):
    """
    Visualizes the training and validation loss and accuracy over epochs.

    Parameters:
    history (History): A History object returned by the fit method of a Keras model. 
                       It contains the training and validation loss and accuracy for each epoch.

    Returns:
    None: Displays the plots for training and validation loss and accuracy.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

    # Plot training and validation loss
    ax1.set_xlabel('Epochs', fontsize=18)
    ax1.set_ylabel('Loss', fontsize=18)
    ax1.plot(history.history['loss'], color='green', label='Training Loss', linewidth=2.5)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], color='orange', linestyle='--', label='Validation Loss', linewidth=2.5)
    ax1.legend(loc='upper right', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Plot training and validation accuracy
    ax2.set_xlabel('Epochs', fontsize=18)
    ax2.set_ylabel('Accuracy', fontsize=18)
    ax2.plot(history.history['accuracy'], color='green', label='Training Accuracy', linewidth=2.5)
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], color='orange', linestyle='--', label='Validation Accuracy', linewidth=2.5)
    ax2.legend(loc='upper left', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    fig.tight_layout()
    plt.show()