import numpy as np
import numpy.random as rng
from sklearn.ensemble import RandomForestClassifier
import dask
import dask.array as da
from dask.distributed import Client, progress
from dask.diagnostics import ProgressBar
from IPython.display import clear_output

from sklearn.model_selection import KFold

def RF(train_data, train_labels, test_data, test_labels):
    """
    Performs cross-validation using RandomForestClassifier on principal components.

    Parameters:
    data (array): Data features.
    labels (array): Data labels.

    Returns:
    list: List of predictions for each fold.
    """

    # Train random forest classifier on first two PCs and apply to test data
    rf = RandomForestClassifier()

    # Train classifier
    rf.fit(train_data, train_labels)

    # Predict unseen test data
    predictions = rf.predict(test_data)

    return predictions, test_labels



# Permutations 
def classification_RF_shuffle(load_train, train_lab_shuffle, load_test, test_lab):
    """
    Trains a RandomForestClassifier with shuffled training labels and evaluates its accuracy on test data.

    Parameters:
    load_train (array): Training data features.
    train_lab_shuffle (array): Training data labels to be shuffled.
    load_test (array): Test data features.
    test_lab (array): Test data labels.

    Returns:
    float: Accuracy of the classifier on the test data.
    """
    
    # Shuffle training labels
    rng = np.random.default_rng()
    rng.shuffle(train_lab_shuffle)

    # Train classifier with shuffled labels
    rf = RandomForestClassifier()
    rf.fit(load_train, train_lab_shuffle)

    # Predict unseen data
    predictions = rf.predict(load_test)

    # compare predictions with test labels and compute accuracy
    result = predictions - test_lab
    result_binary = np.argwhere(result == 0)
    correct = np.size(result_binary,0)
    acc = correct/test_lab.shape[0]*100
    return acc

def iteration_permute(load_train, train_lab_shuffle, load_test, test_lab, n_iterations=10):
    """
    Runs the RandomForest classification with shuffled labels in parallel.
    
    Parameters:
    load_train (array): Training data features.
    train_lab_shuffle (array): Training data labels to be shuffled.
    load_test (array): Test data features.
    test_lab (array): Test data labels.
    n_iterations (int): Number of iterations for shuffling and training.
    
    Returns:
    np.ndarray: Array of accuracy scores from each iteration.
    """

    # Start a Dask distributed client
    client = Client()
    
    # Create delayed tasks
    tasks = [
        dask.delayed(classification_RF_shuffle)(load_train, train_lab_shuffle.copy(), load_test, test_lab)
        for _ in range(n_iterations)
    ]
    
    # Compute tasks in parallel
    results = dask.compute(*tasks)
    acc_shuffle = np.array([float(acc) for acc in results])
    
    client.close() 
    clear_output(wait=False)
    return acc_shuffle



# Principal components
def classification_RF_PCs(train_data, train_lab, test_data, test_lab, loadings, num_pc):
    """
    Trains a RandomForestClassifier using principal components and evaluates its accuracy on test data.

    Parameters:
    train_data (array): Training data features.
    train_lab (array): Training data labels.
    test_data (array): Test data features.
    test_lab (array): Test data labels.
    loadings (array): Principal component loadings.
    num_pc (int): Number of principal components to use.

    Returns:
    float: Accuracy of the classifier on the test data.
    """

    # Project train/test data onto X PC's
    load_train = da.matmul(train_data, loadings[:,:num_pc]).compute()
    load_test = da.matmul(test_data, loadings[:,:num_pc]).compute()
    
    # Train classifier 
    rf = RandomForestClassifier()
    rf.fit(load_train, train_lab)
    
    # Predict unseen data
    predictions = rf.predict(load_test)
    
    # compare predictions with test labels and compute accuracy
    result = predictions - test_lab
    result_binary = np.argwhere(result == 0)
    correct = np.size(result_binary,0)
    acc = correct/test_lab.shape[0]*100
    return acc



def iteration_PCs(train_data, train_lab, test_data, test_lab, loadings, pc_idx):
    """
    Runs the RandomForest classification using principal components in parallel.

    Parameters:
    train_data (array): Training data features.
    train_lab (array): Training data labels.
    test_data (array): Test data features.
    test_lab (array): Test data labels.
    loadings (array): Principal component loadings.
    pc_idx (list): List of principal component indices to use.

    Returns:
    np.ndarray: Array of accuracy scores from each iteration.
    """
    
    # Start a Dask distributed client
    client = Client()

    tasks = [
        dask.delayed(classification_RF_PCs)(train_data, train_lab, test_data, test_lab, loadings, num_pc)
        for num_pc in pc_idx
    ]

    # compute values
    acc_pc = dask.compute(*tasks, scheduler='threads')
    acc_pc = np.array([float(num) for num in acc_pc])
    
    client.close() 
    clear_output(wait = True)
    return acc_pc



# Permutations + principal components
def iteration_permute_PCs(train_data, train_lab_shuffle, test_data, test_lab, loadings, num_pc, n_iterations=10):
    """
    Runs the RandomForest classification with shuffled labels using principal components in parallel.

    Parameters:
    train_data (array): Training data features.
    train_lab_shuffle (array): Training data labels to be shuffled.
    test_data (array): Test data features.
    test_lab (array): Test data labels.
    loadings (array): Principal component loadings.
    num_pc (int): Number of principal components to use.
    n_iterations (int): Number of iterations for shuffling and training.

    Returns:
    np.ndarray: Array of accuracy scores from each iteration.
    """

    # Start a Dask distributed client
    client = Client()

    # Project train/test data onto X PC's
    load_train = da.matmul(train_data, loadings[:,:num_pc]).compute()
    load_test = da.matmul(test_data, loadings[:,:num_pc]).compute()

    # Create delayed tasks
    tasks = [
        dask.delayed(classification_RF_shuffle)(load_train, train_lab_shuffle.copy(), load_test, test_lab)
        for _ in range(n_iterations)
    ]
    
    # Compute tasks in parallel
    results = dask.compute(*tasks)
    acc_pc_shuffle = np.array([float(acc) for acc in results])
    
    client.close() 
    clear_output(wait=False)
    return acc_pc_shuffle