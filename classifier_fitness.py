import numpy as np
import neuralnet
import os.path


def make_te_tr_sets(filename: str):
    """
    Makes the testing and training sets from a larger data set
    :param filename: The filename of the data set to be used
    :return: Two filenames. (training filename, testing filename)
    """
    
    training_fname = filename + '.tr'
    testing_fname = filename + '.te'

    # don't make the files if they already exist
    if os.path.isfile(training_fname) and os.path.isfile(testing_fname):
        return testing_fname, training_fname

    # get the data and shuffle it up
    csv = np.genfromtxt(filename, delimiter=',', dtype=str)
    np.random.shuffle(csv)

    # split it into training and testing
    tr_size = int(.7 * len(csv))
    te_size = len(csv) - tr_size
    tr_data = csv[:tr_size]
    te_data = csv[tr_size:tr_size + te_size]

    print("training_fname", training_fname)
    print("testing_fname", testing_fname)

    # save them out
    np.savetxt(training_fname, tr_data, delimiter=',', fmt='%s')
    np.savetxt(testing_fname, te_data, delimiter=',', fmt='%s')

    # return the filenames
    return testing_fname, training_fname


def get_fitness_file(filename: str, net: neuralnet.FFNN):
    """
    Get the fitness of a neural net for classification
    :param filename: The filename of the data
    :param net: The neural net object
    :return: The fitness score. 0 <= fitness <= 1
    """

    csv = np.genfromtxt(filename, delimiter=",", dtype=str)

    return get_fitness_ds(csv, net)


def get_fitness_ds(csv, net: neuralnet.FFNN):
    numcols = len(csv[0])
    data = csv[:, :numcols - 1]  # the first columns are the data
    targets = csv[:, numcols - 1]  # the last column is the targets

    # pull all the duplicates from the targets list
    target_set = set(targets)
    possible_targets = list(target_set)

    # now run through and test the accuracy
    num_right = 0

    for (row, target) in zip(data, targets):
        outputs = net.get_outputs(row)

        if np.argmax(outputs) == possible_targets.index(target):
            num_right += 1

    # the fitness is the percentage correctly identified
    return num_right / len(data)