import sys
import getopt
import neuralnet
import os
import nnrunner
import numpy as np
import copy

# constants
NET_INPUTS = 5
NET_OUTPUTS = 3
INIT_DIR = "generationinit"
NUM_AVERAGE = 3


def main(argv):

    # How the program is to be used
    usage = "\tusage: --pop=[population size] --it=[number of iterations] --topology=[t1,t2] --inertia=[w]" \
            " --c1=[c1] --c2=[c2]\n"

    optlist, args = getopt.getopt(argv[1:], "h", ["help", "pop=", "it=", "topology=", "inertia=", "c1=", "c2="])

    # default parameters
    population_size = 100
    num_iterations = 100
    topology = []
    w = .7968
    c1 = 1.4962
    c2 = 1.4962

    # replace the parameters
    for key, val in optlist:
        if key in ("-h", "--help"):
            print(usage)
            return
        elif key == "--pop":
            population_size = int(val)
        elif key == "--it":
            num_iterations = int(val)
        elif key == "--topology":
            entries = val.split(",")
            for entry in entries:
                topology.append(int(entry))
        elif key == "--inertia":
            w = float(val)
        elif key == "c1":
            c1 = float(val)
        elif key == "c2":
            c2 = float(val)

    # can't run without a topology
    if not topology:
        print("Topology must be specified. Use -h or --help for help")

    # init the global best
    g_best = {'fitness': 0}

    brains = generate_brains(population_size, topology, g_best)

    # open the points file
    points = open("points.csv", "w")
    points.truncate()

    for iteration in range(0, num_iterations):
        print("Generation " + str(iteration + 1))

        # make the directory if it doesn't exist
        dirname = "generation" + str(iteration + 1)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # the index file
        index = open(dirname + "/index.csv", "w")

        # update particles
        update_particles(brains, g_best, w, c1, c2)

        for organism in brains:
            # write to file
            organismfilename = dirname + "/" + organism['name']
            neuralnet.to_file(organismfilename, organism['net'])

            # write out to the index and the points
            index.write(str(organism['fitness']) + "," + organismfilename.split('/')[-1] + "\n")
            points.write(str(iteration) + "," + str(organism['fitness']) + "\n")

        index.close()

    points.close()


def update_particles(population, g_best, w, c1, c2):

    for organism in population:

        for i in range(0, len(organism['net'].layers)):
            for j in range(0, len(organism['net'].layers[i])):
                for k in range(0, len(organism['net'].layers[i][j])):

                    # update velocity
                    # v(t + 1) = w * v(t) + c1 * (p(t) - x(t)) + c2 * (g(t) - x(t))
                    organism['velocity'].layers[i][j][k] = w * organism['velocity'].layers[i][j][k] \
                                    + c1 * (organism['best_net'].layers[i][j][k] - organism['net'].layers[i][j][k]) \
                                    + c2 * (g_best['net'].layers[i][j][k] - organism['net'].layers[i][j][k])

                    # update position
                    # x(t + 1) = x(t) + v(t + 1)
                    organism['net'].layers[i][j][k] += organism['velocity'][i][j][k]

                    # evaluate and check against the global best and personal best
                    neuralnet.to_file('temp.net', organism['net'])
                    organism['fitness'] = get_fitness('temp.net')

                    if organism['fitness'] > organism['best_fitness']:
                        organism['best_net'] = copy.deepcopy(organism['net'])
                        organism['best_fitness'] = organism['fitness']


def generate_brains(population, topology, g_best):
    brains = []

    # make a directory
    if not os.path.exists(INIT_DIR):
        os.makedirs(INIT_DIR)

    for i in range(0, population):
        organism = {
            # TODO may need to start with a wider range of weights
            'net': neuralnet.FFNN(topology, NET_INPUTS, NET_OUTPUTS),
            'name': "0-" + str(i) + ".net",
            'velocity': neuralnet.FFNN(topology, NET_INPUTS, NET_OUTPUTS)}

        # start with zero velocity
        neuralnet.zero_out(organism['velocity'])

        # write it out and evaluate
        neuralnet.to_file(INIT_DIR + "/" + organism['name'], organism['net'])
        organism['fitness'] = get_fitness(INIT_DIR + "/" + organism['name'])

        # best is now
        organism['best_net'] = copy.deepcopy(organism['net'])
        organism['best_fitness'] = organism['fitness']

        # start the best with the best
        if organism['fitness'] > g_best['fitness']:
            g_best['net'] = copy.deepcopy(organism['net'])

        brains.append(organism)

    return brains


def get_fitness(fname):
    scores = []
    for i in range(0, NUM_AVERAGE):
        scores.append(nnrunner.run(fname))
    return np.mean(scores)


# This is here to ensure main is only called when
#   this file is run, not just loaded
if __name__ == "__main__":
    main(sys.argv)
