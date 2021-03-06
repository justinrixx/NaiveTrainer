"""
A neural net is just a list of lists of lists. (layers, nodes, weights).
So this file is just a collection of functions that are useful for manipulating
that type of a structure.
"""

import random
import numpy as np


class FFNN:
    """ A multi-layer feed-forward neural net """

    def __init__(self, topology, num_inputs, num_outputs):
        self.topology = topology
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = make_net(topology, num_inputs, num_outputs)

    def get_outputs(self, inputs):
        """
        Get the outputs of a neural network from the inputs
        :param inputs: A 1-dimensional array of numbers to input to the net
        :return: A 1-dimensional array of numbers that the net outputs
        """

        # first layer
        out = get_outputs(self.layers[0], inputs)

        for i in range(1, len(self.layers)):
            out = get_outputs(self.layers[i], out)

        return out


def make_net(topology, num_inputs, num_outputs):
    """
    Makes a neural net (list of lists of lists) that is the desired size, and fills
    it with random values
    """

    # input layer
    layers = [make_layer(num_inputs, topology[0])]

    # hidden layers
    for i in range(1, len(topology)):
        layers.append(make_layer(len(layers[i - 1]), topology[i]))

    # output layer
    layers.append(make_layer(len(layers[-1]), num_outputs))

    return layers


def make_layer(num_inputs, num_nodes):
    """
    Makes a layer of nodes in a neural net
    """

    layer = []

    for i in range(0, num_nodes):
        layer.append(make_node(num_inputs))

    return layer


def make_node(num_inputs):
    """
    Makes a node in a neural network
    """

    node = []

    # +1 for the bias
    for i in range(0, num_inputs + 1):
        weight = np.random.exponential()

        # 50% chance to be negative
        if random.randint(0, 1) == 0:
            weight *= -1

        node.append(weight)

    return node


def sp_crossover(net1, net2):
    """
    Single Point Crossover
    Picks a random point to split on, and the offspring get net1's DNA up to the
    split point, and net2's DNA after the split point. A second offspring is produced
    that is the opposite of the first
    :param net1: The first parent
    :param net2: The second parent
    :return: two offspring, each the opposite of the other
    """

    total = 0
    for layer in net1.layers:
        for node in layer:
            total += len(node)

    # the babies that will be returned
    child1 = FFNN(net1.topology, net1.num_inputs, net1.num_outputs)
    child2 = FFNN(net1.topology, net1.num_inputs, net1.num_outputs)

    split = random.randint(1, total - 1)

    index = 0
    for i in range(0, len(net1.layers)):
        for j in range(0, len(net1.layers[i])):
            for k in range(0, len(net1.layers[i][j])):
                if index < split:
                    child1.layers[i][j][k] = net1.layers[i][j][k]
                    child2.layers[i][j][k] = net2.layers[i][j][k]
                else:
                    child2.layers[i][j][k] = net1.layers[i][j][k]
                    child1.layers[i][j][k] = net2.layers[i][j][k]

                index += 1

    return child1, child2


def u_crossover(net1, net2):
    """
    Uniform Crossover. For each weight, a parent is chosen at random, and the
    DNA is taken from that parent.
    :param net1: The first parent
    :param net2: The second parent
    :return: two offspring, each the opposite of the other
    """

    child1 = FFNN(net1.topology, net1.num_inputs, net1.num_outputs)
    child2 = FFNN(net1.topology, net1.num_inputs, net1.num_outputs)

    for i in range(0, len(net1.layers)):
        for j in range(0, len(net1.layers[i])):
            for k in range(0, len(net1.layers[i][j])):
                if random.randint(0, 1) == 0:
                    child1.layers[i][j][k] = net1.layers[i][j][k]
                    child2.layers[i][j][k] = net2.layers[i][j][k]
                else:
                    child2.layers[i][j][k] = net1.layers[i][j][k]
                    child1.layers[i][j][k] = net2.layers[i][j][k]

    return child1, child2


def to_file(filename, net):
    """
    Write a neural net to a file
    """
    if type(net) != FFNN:
        raise TypeError('Cannot write an object that is not a FFNN to a file')

    file = open(filename, "w")
    file.truncate()

    # num inputs, num outputs, num hidden layers
    contents = str(net.num_inputs) + " " + str(net.num_outputs) + " " + str(len(net.layers) - 1)

    # topology
    for i in net.topology:
        contents += " " + str(i)

    # weights
    for i in range(0, len(net.layers)):
        for j in range(0, len(net.layers[i])):
            for weight in net.layers[i][j]:
                contents += " " + str(weight)

    file.write(contents)
    file.close()


def from_file(filename):
    """
    Read a neural net from a file
    :return: The neural net that was read
    """

    # ['4', '3', '1', '4', '-0.8267162400009331', ... ]
    l = list(np.genfromtxt(filename, dtype=str, delimiter=' '))
    assert(len(l) > 4)

    num_inputs = int(l[0])
    num_outputs = int(l[1])
    i_read = 2

    topology = []
    for i in range(0, int(l[2])):
        topology.append(int(l[i_read]))
        i_read += 1
    assert(len(topology) > 0)

    # input layer
    net = FFNN(topology, num_inputs, num_outputs)
    input_layer = []

    for i in range(0, topology[0]):

        node = []
        for j in range(0, num_inputs + 1):
            node.append(float(l[i_read]))
            i_read += 1

        input_layer.append(node)

    net.layers[0] = input_layer

    # hidden layers
    for i in range(1, len(topology)):

        layer = []
        for j in range(0, topology[i]):

            node = []
            for k in range(0, len(net.layers[i - 1])):
                node.append(float(l[i_read]))
                i_read += 1
            layer.append(node)
        net.layers[i] = layer

    return net


def mutate(net):
    """
    Mutates a neural net in place
    :type net: The FFNN object or the list of layers. Either one
    """

    if type(net) == FFNN:
        network = net.layers
    elif type(net) == list:
        network = net
    else:
        raise TypeError('Net is not a FFNN or list')

    # The chance for each weight to change is 1 / L where L is
    #   the number of weights in the network
    #L = 0
    #for i in range(0, len(network)):
    #    for j in range(0, len(network[i])):
    #        L += len(network[i][j])
    L = 10

    for i in range(0, len(network)):
        for j in range(0, len(network[i])):
            for k in range(0, len(network[i][j])):

                coin = random.randint(0, L)
                if coin == 1:
                    # -5 <= r <= 5
                    #r = (random.random() - .5) * 5.0
                    r = np.random.exponential()

                    # 50% chance to be negative
                    if random.randint(0, 1) == 0:
                        r *= -1

                    network[i][j][k] += r


def get_outputs(layer, inputs):
    """
    A helper function that gets the output of a layer in a neural net
    :param layer: A 2-dimensional array of numbers ( layer = [node = [ weights ] ])
    :param inputs: A 1-dimensional array of numbers. Inputs to the layer
    :return: The output of the layer
    """

    outputs = []

    for node in layer:
        # there should be 1 extra input for the bias
        assert len(node) == len(inputs) + 1

        total = 0
        for i in range(0, len(node) - 1):
            total += node[i] * float(inputs[i])

        # bias
        total += node[-1] * -1
        outputs.append(np.tanh(total))

    return outputs


def zero_out(net: FFNN):
    """
    Sets all the weights in a neural net to 0
    :param net: The neural net to zero out
    """

    for i in range(0, len(net.layers)):
        for j in range(0, len(net.layers[i])):
            for k in range(0, len(net.layers[i][j])):
                net.layers[i][j][k] = 0
