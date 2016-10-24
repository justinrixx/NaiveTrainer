import sys
import getopt
import neuralnet
import os
import nnrunner
import numpy as np

# constants
NET_INPUTS = 31
NET_OUTPUTS = 4
INIT_DIR = "generationinit"


def main(argv):

    # How the program is to be used
    usage = "\tusage: --pop=[population size] --it=[number of iterations] --surv=[chance to survive]" \
            " --topology=[layer1,layer2,etc]\n"

    optlist, args = getopt.getopt(argv[1:], "h", ["help", "pop=", "it=", "surv=", "topology="])

    # default parameters
    population_size = 100
    num_iterations = 100
    survival_percentage = .3
    topology = []

    # replace the parameters
    for key, val in optlist:
        if key in ("-h", "--help"):
            print(usage)
            return
        elif key == "--pop":
            population_size = int(val)
        elif key == "--it":
            num_iterations = int(val)
        elif key == "--surv":
            survival_percentage = float(val)
        elif key == "--topology":
            entries = val.split(",")
            for entry in entries:
                topology.append(int(entry))

    # can't run without a topology
    if not topology:
        print("Topology must be specified. Use -h or --help for help")

    cutoff_point = int(survival_percentage * population_size)

    brains = generate_brains(population_size, topology)

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

        # life is hard . . .
        kill_stuff(brains, cutoff_point)
        repopulate(brains, population_size, iteration)

        for organism in brains:
            # write to file
            organismfilename = dirname + "/" + organism['name']
            neuralnet.to_file(organismfilename, organism['net'])

            # evaluate
            score = nnrunner.run(organismfilename)

            organism['fitness'] = int((((organism['gen'] - 1) / organism['gen']) * organism['fitness'])
                                      + (1 / organism['gen'] * score))

            # write out to the index and the points
            index.write(str(organism['fitness']) + "," + organismfilename.split('/')[-1] + "\n")
            points.write(str(iteration) + "," + str(organism['fitness']) + "\n")

        index.close()

    points.close()


def generate_brains(population, topology):
    brains = []

    # make a directory
    if not os.path.exists(INIT_DIR):
        os.makedirs(INIT_DIR)

    for i in range(0, population):
        organism = {
            'net': neuralnet.FFNN(topology, NET_INPUTS, NET_OUTPUTS),
            'name': "0-" + str(i) + ".net",
            'gen': 2}

        # write it out and evaluate
        neuralnet.to_file(INIT_DIR + "/" + organism['name'], organism['net'])
        organism['fitness'] = nnrunner.run(INIT_DIR + "/" + organism['name'])

        brains.append(organism)

    return brains


def kill_stuff(brains, cutoff):

    # http://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-values-of-the-dictionary-in-python
    # sort it out, strongest at the front
    brains.sort(key=lambda k: k['fitness'], reverse=True)

    # kill the ones that don't deserve to live
    for i in range(0, len(brains) - cutoff):
        brains.pop()


def repopulate(brains, population, generation):

    orgnum = 0

    diff = population - len(brains)

    # make the fitnesses into a probability distribution
    fits = [x['fitness'] for x in brains]
    dist = [x / np.sum(fits) for x in fits]

    parents = np.random.choice(brains, size=(diff * 2), p=dist)

    for i in range(0, len(parents), 2):

        child1, child2 = neuralnet.sp_crossover(parents[i]['net'], parents[i + 1]['net'])

        neuralnet.to_file("temp.net", child1)
        score1 = nnrunner.run("temp.net")

        neuralnet.to_file("temp.net", child2)
        score2 = nnrunner.run("temp.net")

        organism = {'name': str(generation + 1) + "-" + str(orgnum) + ".net",
                    'gen': 2}

        # only the strong are cared for
        if score1 > score2:
            organism['fitness'] = score1
            organism['net'] = child1
        else:
            organism['fitness'] = score1
            organism['net'] = child1

        brains.append(organism)
        orgnum += 1


# This is here to ensure main is only called when
#   this file is run, not just loaded
if __name__ == "__main__":
    main(sys.argv)
