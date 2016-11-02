import numpy as np
import sys


def main(argv):

    if len(argv) != 2:
        usage(argv[0])
        return

    csv = np.genfromtxt(argv[1], delimiter=",", dtype=str)
    numcols = len(csv[0])
    target_set = set(csv[:, numcols - 1])

    numattrs = len(csv[0]) - 1
    numtargets = len(target_set)
    numinstances = len(csv)

    print('Dataset at ' + argv[1] + ' has the following attributes:')
    print('\t', numattrs, ' inputs')
    print('\t', numtargets, ' outputs')
    print('\t', numinstances, ' instances')


def usage(program_name):
    print('\tusage: python ' + program_name + ' dataset.csv')


if __name__ == "__main__":
    main(sys.argv)
