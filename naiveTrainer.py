import sys
import getopt


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

    print("all the options:\npopulation_size: " + str(population_size)
          + "\nnum_iterations: " + str(num_iterations)
          + "\nsurvival_percentage: " + str(survival_percentage)
          + "\ntopology:", topology)

    return 0


# This is here to ensure main is only called when
#   this file is run, not just loaded
if __name__ == "__main__":
    main(sys.argv)
