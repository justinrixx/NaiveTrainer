"""
A neural net is just a list of lists of lists. (layers, nodes, weights).
So this file is just a collection of functions that are useful for manipulating
that type of a structure.
"""

import random


def make_net(topology):
    """
    Makes a neural net (list of lists of lists) that is the desired size, and fills
    it with random values
    """

    return []


def sp_crossover(net1, net2, topology):
    """
    Single Point Crossover
    Picks a random point to split on, and the offspring get net1's DNA up to the
    split point, and net2's DNA after the split point. A second offspring is produced
    that is the opposite of the first
    :param net1: The first parent
    :param net2: The second parent
    :param topology: The topology of the neural net
    :return: two offspring, each the opposite of the other
    """

    total = 0
    for layer in net1:
        for node in layer:
            total += len(node)

    # the babies that will be returned
    child1 = make_net(topology)
    child2 = make_net(topology)

    split = random.randint(1, total - 1)

    index = 0
    for i in range(0, len(net1)):
        for j in range(0, len(net1[i])):
            for k in range(0, len(net1[i][j])):
                if index < split:
                    child1[i][j][k] = net1[i][j][k]
                    child2[i][j][k] = net2[i][j][k]
                else:
                    child2[i][j][k] = net1[i][j][k]
                    child1[i][j][k] = net2[i][j][k]

                index += 1

    return child1, child2


def u_crossover(net1, net2, topology):
    """
    Uniform Crossover. For each weight, a parent is chosen at random, and the
    DNA is taken from that parent.
    :param net1: The first parent
    :param net2: The second parent
    :param topology: The topology of the neural net
    :return: two offspring, each the opposite of the other
    """

    child1 = make_net(topology)
    child2 = make_net(topology)

    for i in range(0, len(net1)):
        for j in range(0, len(net1[i])):
            for k in range(0, len(net1[i][j])):
                if random.randint(0, 1) == 0:
                    child1[i][j][k] = net1[i][j][k]
                    child2[i][j][k] = net2[i][j][k]
                else:
                    child2[i][j][k] = net1[i][j][k]
                    child1[i][j][k] = net2[i][j][k]

    return child1, child2


def to_file(filename, net):
    """
    TODO write a neural net to a file
    """
    return 0


def mutate(net):
    """
    TODO mutates a neural net in place
    """
