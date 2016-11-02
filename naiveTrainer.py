import sys
import getopt
import neuralnet
import os
import nnrunner
import classifier_fitness as cf
import numpy as np

# constants
NET_INPUTS = 4
NET_OUTPUTS = 3
INIT_DIR = "generationinit"
NUM_AVERAGE = 3
DATASET_FNAME = "/home/justin/Downloads/iris.csv"

# globals because yeah
dataset = DATASET_FNAME
num_inputs = NET_INPUTS
num_outputs = NET_OUTPUTS


def main(argv):

    # How the program is to be used
    usage = "\tusage: --pop=[population size] --it=[number of iterations] --surv=[chance to survive]" \
            " --topology=[layer1,layer2,etc], --ds=[dataset filename], --in[# of inputs] --out[# of outputs]\n"

    optlist, args = getopt.getopt(argv[1:], "h", ["help", "pop=", "it=", "surv=",
                                                  "topology=", "ds=", "in=", "out="])

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
        elif key == "--ds":
            dataset = str(val)
        elif key == "--in":
            num_inputs = int(val)
        elif key == "--out":
            num_outputs = int(val)

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

        # kill stuff
        fits = [x['fitness'] for x in brains]
        dist = [x / np.sum(fits) for x in fits]
        brains = list(np.random.choice(brains, size=cutoff_point, p=dist, replace=False))

        repopulate(brains, population_size, iteration)

        for organism in brains:
            # write to file
            organismfilename = dirname + "/" + organism['name']
            neuralnet.to_file(organismfilename, organism['net'])

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
            'net': neuralnet.FFNN(topology, num_inputs, num_outputs),
            'name': "0-" + str(i) + ".net",
            'gen': 2}

        # write it out and evaluate
        #neuralnet.to_file(INIT_DIR + "/" + organism['name'], organism['net'])
        #organism['fitness'] = get_fitness(INIT_DIR + "/" + organism['name'])
        organism['fitness'] = get_fitness(dataset, organism['net'])

        brains.append(organism)

    return brains


def repopulate(brains, population, generation):

    orgnum = 0

    diff = population - len(brains)

    # make the fitnesses into a probability distribution
    fits = [x['fitness'] for x in brains]
    dist = [x / np.sum(fits) for x in fits]

    parents = np.random.choice(brains, size=(diff * 2), p=dist)

    for i in range(0, len(parents), 2):

        child1, child2 = neuralnet.sp_crossover(parents[i]['net'], parents[i + 1]['net'])
        #child1, child2 = neuralnet.u_crossover(parents[i]['net'], parents[i + 1]['net'])

        #neuralnet.to_file("temp.net", child1)
        #score1 = get_fitness("temp.net")
        score1 = get_fitness(dataset, child1)

        #neuralnet.to_file("temp.net", child2)
        #score2 = get_fitness("temp.net")
        score2 = get_fitness(dataset, child2)

        organism = {'name': str(generation + 1) + "-" + str(orgnum) + ".net",
                    'gen': 2}

        # only the strong are cared for
        if score1 > score2:
            organism['fitness'] = score1
            organism['net'] = child1
        else:
            organism['fitness'] = score2
            organism['net'] = child2

        brains.append(organism)
        orgnum += 1


#def get_fitness(fname):
#    scores = []
#    for i in range(0, NUM_AVERAGE):
#        scores.append(nnrunner.run(fname))
#    return int(np.mean(scores))


def get_fitness(ds, net):
    return cf.get_fitness_file(ds, net)


# This is here to ensure main is only called when
#   this file is run, not just loaded
if __name__ == "__main__":
    main(sys.argv)
