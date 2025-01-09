import numpy as np
import numpy.random as rng
from sklearn.ensemble import RandomForestClassifier
import dask

def classification_RF_shuffle(load_train, train_lab_shuffle, load_test, test_lab):
    
    rng = np.random.default_rng()

    # Train classifier with new shuffled labels each iteration
    rf = RandomForestClassifier()
    rng.shuffle(train_lab_shuffle)
    rf.fit(load_train, train_lab_shuffle)

    # Predict unseen data
    predictions = rf.predict(load_test)

    # compare predictions with test labels and compute accuracy
    result = predictions - test_lab
    result_binary = np.argwhere(result == 0)
    correct = np.size(result_binary,0)
    acc = correct/test_lab.shape[0]*100
    return acc


def iteration_permute(load_train, train_lab_shuffle, load_test, test_lab):
    
    acc_shuffle = []
    # limit number of iterations to 100 for memory considerations
    for i in range(100):
        acc_tmp = dask.delayed(classification_RF_shuffle)(load_train, train_lab_shuffle, load_test, test_lab)
        acc_shuffle.append(acc_tmp)
    
    # compute values
    acc_shuffle = dask.compute(*acc_shuffle)
    acc_shuffle = np.array([float(num) for num in acc_shuffle])
    clear_output(wait = True)

    return acc_shuffle