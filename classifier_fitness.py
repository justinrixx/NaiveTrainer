import numpy as np
import neuralnet
import time


def get_fitness_file(filename: str, net: neuralnet.FFNN):
    """
    Get the fitness of a neural net for classification
    :param filename: The filename of the data
    :param net: The neural net object
    :return: The fitness score. 0 <= fitness <= 1
    """

    csv = np.genfromtxt(filename, delimiter=",", dtype=str)
    numcols = len(csv[0])
    data = csv[:, :numcols - 1]  # the first columns are the data
    targets = csv[:, numcols - 1]  # the last column is the targets

    # test with 70% to avoid over-fitting
    cutoff_point = int(.7 * len(csv))

    # get a random sample of the data
    seed = int(time.time())
    np.random.seed(seed)
    data = data[np.random.choice(data.shape[0], size=cutoff_point, replace=False)]
    np.random.seed(seed)
    targets = np.random.choice(targets, size=cutoff_point)

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
