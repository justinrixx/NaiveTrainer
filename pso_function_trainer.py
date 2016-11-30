import sys
import getopt
import os
import numpy as np
import copy

# constants
INIT_DIR = "generationinit"
DIMENSIONS = 7


def main(argv):

    # How the program is to be used
    usage = "\tusage: --pop=[population size] --it=[number of iterations] --inertia=[w]" \
            " --c1=[c1] --c2=[c2]\n"

    optlist, args = getopt.getopt(argv[1:], "h", ["help", "pop=", "it=", "inertia=",
                                                  "c1=", "c2="])

    # default parameters
    population_size = 100
    num_iterations = 100
    w = 1
    wdamp = .99
    c1 = 2
    c2 = 2

    # replace the parameters
    for key, val in optlist:
        if key in ("-h", "--help"):
            print(usage)
            return
        elif key == "--pop":
            population_size = int(val)
        elif key == "--it":
            num_iterations = int(val)
        elif key == "--inertia":
            w = float(val)
        elif key == "--c1":
            c1 = float(val)
        elif key == "--c2":
            c2 = float(val)

    # init the global best
    g_best = {'fitness': 999999999}

    brains = generate_brains(population_size, g_best)

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
            #neuralnet.to_file(organismfilename, organism['net'])

            # write out to the index and the points
            index.write(str(organism['fitness']) + "," + organismfilename.split('/')[-1] + "\n")

        index.close()

        points.write(str(iteration) + "," + str(g_best['fitness']) + "\n")

        w *= wdamp

    points.close()


def update_particles(population, g_best, w, c1, c2):

    for organism in population:

        organism['velocity'] = (organism['velocity'] * w) + \
                               ((organism['best_net'] - organism['net']) * c1 * np.random.random()) + \
                               ((g_best['net'] - organism['net']) * c2 * np.random.random())
        organism['net'] = organism['net'] + organism['velocity']

        # evaluate and check against the global best and personal best
        organism['fitness'] = get_fitness(organism['net'])

        # update the personal best
        if organism['fitness'] < organism['best_fitness']:
            #print("Updating personal best from " + str(organism['best_fitness']) + " to " + str(organism['fitness']))
            organism['best_net'] = copy.deepcopy(organism['net'])
            organism['best_fitness'] = organism['fitness']

            # update the global best -- only check if new personal best
            if organism['fitness'] < g_best['fitness']:
                #print("Updating global best from " + str(g_best['fitness']) + " to " + str(organism['fitness']))
                g_best['net'] = copy.deepcopy(organism['net'])
                g_best['fitness'] = organism['fitness']


def generate_brains(population, g_best):
    """
    Generate an array of random vectors. They are from (-100, 100)
    :param population: The number of organisms to generate
    :param g_best: The globally best solution
    :return: The list of brains
    """
    brains = []

    # make a directory
    if not os.path.exists(INIT_DIR):
        os.makedirs(INIT_DIR)

    for i in range(0, population):
        organism = {
            'net': np.random.rand(DIMENSIONS) * 200,
            'name': "0-" + str(i) + ".net",
            'velocity': np.zeros(DIMENSIONS)}

        # evaluate
        organism['fitness'] = get_fitness(organism['net'])

        # best is now
        organism['best_net'] = copy.deepcopy(organism['net'])
        organism['best_fitness'] = organism['fitness']

        # start the best with the best
        if organism['fitness'] < g_best['fitness']:
            #print("Updating global best from " + str(g_best['fitness']) + " to " + str(organism['fitness']))
            g_best['net'] = copy.deepcopy(organism['net'])
            g_best['fitness'] = organism['fitness']

        brains.append(organism)

    return brains


def get_fitness(net):
    return np.sum(net ** 2)


# This is here to ensure main is only called when
#   this file is run, not just loaded
if __name__ == "__main__":
    main(sys.argv)
