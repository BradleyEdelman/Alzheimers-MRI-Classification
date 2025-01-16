import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import numpy as np
import seaborn as sns


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



def multiclass_summary(predictions, true_labels, class_lab):
    """
    Summarizes the performance of a multiclass classification model.

    Parameters:
    predictions (list of arrays): List of arrays containing the predicted labels for each sample.
    true_labels (list of arrays): List of arrays containing the true labels for each sample.
    class_lab (list of str): List of class labels.

    Returns:
    None: Displays the confusion matrix and class-specific ROC curves, and prints precision, recall, and F1 scores for each class.
    """
    
    # Probabilities or absolute classes
    if len(predictions.shape) == 2:
        predictions_cm = np.argmax(predictions, axis=1)
    else:
        predictions_cm = predictions

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Generate and plot confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions_cm)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label': 'Accuracy (%)'}, 
                xticklabels=class_lab, yticklabels=class_lab, ax=ax1, linewidths=1, linecolor='black', annot_kws={"size": 10}, vmin=0, vmax=100)
    ax1.set_xlabel('Predicted Labels', fontsize=10)
    ax1.set_ylabel('True Labels', fontsize=10)
    ax1.set_title('Confusion Matrix', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Plot class-specific ROC curves
    for i, label in enumerate(class_lab):

        if len(predictions.shape)==2:
            fpr, tpr, _ = roc_curve(true_labels == i, predictions[:, i])
        else:
            fpr, tpr, _ = roc_curve(true_labels == i, predictions == i)

        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=8)
    ax2.set_ylabel('True Positive Rate', fontsize=8)
    ax2.set_title('Class-specific ROC Curves', fontsize=10)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)

    fig.tight_layout()
    plt.show()

    # Print precision, recall, and F1 scores
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions_cm, average=None, labels=range(len(class_lab)))
    for i, label in enumerate(class_lab):
        print(f'{label} - Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1 Score: {f1[i]:.2f}')












