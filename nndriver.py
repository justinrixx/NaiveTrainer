import sys
import neuralnet


def main(argv):

    print("""Neural Net Tester. Input your choice:
    1) Mutation
    2) spCrossover
    3) uCrossover
    4) Save to file""")

    choice = int(input())

    if choice > 3 or choice < 1:
        print("Invalid choice")
        return

    # Mutation
    if choice == 1:
        net = neuralnet.FFNN([3, 3, 3], 3, 3)
        print("Original network\n", net.layers, "\n")

        neuralnet.mutate(net)
        print("Mutated network\n", net.layers, "\n")

    # Single point crossover
    elif choice == 2:
        net1 = neuralnet.FFNN([3, 3, 3], 3, 3)
        net2 = neuralnet.FFNN([3, 3, 3], 3, 3)

        print("Parent 1\n", net1.layers, "\n")
        print("Parent 2\n", net2.layers, "\n")

        child1, child2 = neuralnet.sp_crossover(net1, net2)
        print("Child 1", child1.layers)
        print("Child 2", child2.layers)

    # Uniform crossover
    elif choice == 3:
        net1 = neuralnet.FFNN([3, 3, 3], 3, 3)
        net2 = neuralnet.FFNN([3, 3, 3], 3, 3)

        print("Parent 1", net1.layers)
        print("Parent 2", net2.layers)

        child1, child2 = neuralnet.u_crossover(net1, net2)
        print("Child 1", child1.layers)
        print("Child 2", child2.layers)

    elif choice == 4:
        net = neuralnet.FFNN([3, 3, 3], 3, 3)
        print("Original network", net.layers)
        print("Writing to test.net")
        neuralnet.to_file("test.net", net)

# This is here to ensure main is only called when
#   this file is run, not just loaded
if __name__ == "__main__":
    main(sys.argv)
