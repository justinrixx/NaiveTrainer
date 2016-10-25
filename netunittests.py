import neuralnet
import sys


def main(argv):
    """
    This program runs a suite of unit tests
    """
    test_net_sizes()


def test_net_sizes():
    """
    Tests whether neural nets that are created are of the correct size
    """

    # Test 1
    net = neuralnet.FFNN([2, 2, 3, 3], 3, 3)
    expected_layers = [2, 2, 3, 3, 3]
    expected_weights = [[4, 4], [3, 3], [3, 3, 3], [4, 4, 4], [4, 4, 4]]
    expected_weight_number = 47

    test_net_size(net, expected_layers, expected_weights, expected_weight_number)

    # Test 2 - the size I've been playing with
    net = neuralnet.FFNN([18], 31, 4)
    expected_layers = [18, 4]
    # 18 32s, 4 19s
    expected_weights = [[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
                        [19, 19, 19, 19]]
    expected_weight_number = 652

    test_net_size(net, expected_layers, expected_weights, expected_weight_number)


def test_net_size(net, expected_layers, expected_weights, expected_weight_number):
    """
    Tests the size of a neural net to make sure it's correct
    :param net: The neural net. Must be a FFNN object
    :param expected_layers: A 1-dimensional list of the expected size of each layer
    :param expected_weights: A 2-dimensional list of the expected number of weights per node
    [layer][node]
    :param expected_weight_number: The expected total number of weights in the network
    """
    assert len(net.layers) == len(expected_layers), \
        'len(net.layers): %(act)d, len(expected_layers): %(exp)d' \
        % {'act': len(net.layers), 'exp': len(expected_layers)}

    total = 0
    for i in range(0, len(expected_layers)):
        # layer should have the expected number of nodes
        assert len(net.layers[i]) == expected_layers[i], \
            'len(net.layers[%(i)d]): %(act)d, expected_layers[i]: %(exp)d' \
            % {'act': len(net.layers[i]), 'exp': expected_layers[i], 'i': i}

        for j in range(0, len(net.layers[i])):
            # node should have the expected number of weights
            assert len(net.layers[i][j]) == expected_weights[i][j], \
                'len(net.layers[%(i)d][%(j)d]): %(act)d, expected_weights[%(i)d][%(i)d]: %(exp)d' \
                % {'act': len(net.layers[i][j]), 'exp': expected_weights[i][j], 'i': i, 'j': j}
            total += len(net.layers[i][j])

    # network as a whole should have the correct number of weights
    assert total == expected_weight_number, 'total: %(act)d, expected_weight_number: %(exp)d' \
        % {'exp': expected_weight_number, 'act': total}


if __name__ == "__main__":
    main(sys.argv)
