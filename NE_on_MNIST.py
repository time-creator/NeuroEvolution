import numpy
import torch
import autoencoder as ae
import random as rnd
import copy

# TODO: delete population class, instead just have list

# function to input into the needed format
def clean_input(datasheet_path):
    test_data_file = open(f"{datasheet_path}", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    inputs = []
    targets = []
    # I want pairs: 784 pixel values, target
    for record in test_data_list:
        all_values = record.split(',')
        input = torch.zeros(28 * 28, dtype=torch.double)
        temp_var = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        input = torch.add(input, torch.from_numpy(temp_var)).float()
        target = int(all_values[0])
        inputs.append(input)
        targets.append(target)

    return inputs, targets


# populates a population
def populate(population_size):
    population = []
    for i in range(population_size):
        population.append(ae.EvolutionNet())
    return population


# creates a new population
class nnPopulation:

    def __init__(self, population_size):
        self.population_size = population_size
        self.population = populate(population_size)


def fitness(network, inputs, targets):
    score = 0
    for input, target in zip(inputs, targets):
        score += network.forward(input)[target].item()
    score /= len(inputs)

    return score


def evaluate_population(population, inputs, targets):
    results = []
    for network in population:
        results.append(fitness(network, inputs, targets))
    return results


def evolve(population, train_inputs, train_targets, mutation_rate=0.03,):
    """ First we apply the survival of the fittest principle """
    nets_and_results = list(zip(population, evaluate_population(population, train_inputs, train_targets)))
    nets_and_results.sort(key=lambda x: x[1], reverse = True)
    del nets_and_results[2:]
    # TODO: do not hardcode to a population size of 10
    # bring the list back up to 10 elements
    filler = []
    filler.append(copy.deepcopy(nets_and_results[0]))
    filler.append(copy.deepcopy(nets_and_results[1]))
    nets_and_results.extend(copy.deepcopy(filler))
    nets_and_results.extend(copy.deepcopy(filler))
    nets_and_results.extend(copy.deepcopy(filler))
    nets_and_results.extend(copy.deepcopy(filler))

    """ After that we mutate the fittest individuals """
    for net_and_result in nets_and_results:
        mutation_vector_fc1 = torch.zeros(net_and_result[0].fc1.weight.data.size())
        for item in mutation_vector_fc1:
            item.add_((numpy.random.normal(0.0000, 0.002) if numpy.random.random() < mutation_rate else 0))
        net_and_result[0].fc1.weight.data = net_and_result[0].fc1.weight.data.add(mutation_vector_fc1)

        mutation_vector_fc2 = torch.zeros(net_and_result[0].fc2.weight.data.size())
        for item in mutation_vector_fc2:
            item.add_((numpy.random.normal(0.0000, 0.005) if numpy.random.random() < mutation_rate else 0))
        net_and_result[0].fc2.weight.data = net_and_result[0].fc2.weight.data.add(mutation_vector_fc2)

    new_population, _ = zip(*nets_and_results)
    return new_population


def main():

    test_pop = nnPopulation(10)
    train_inputs, train_targets = clean_input(datasheet_path="D:/workFolder/NeuroEvolution/mnist_dataset/mnist_train_100.txt")
    validate_inputs, validate_targets = clean_input(datasheet_path="D:/workFolder/NeuroEvolution/mnist_dataset/mnist_test_10.txt")
    number_of_generations = 100

    print(evaluate_population(test_pop.population, train_inputs, train_targets))
    gen_i = test_pop.population # = gen_0 aka initial population
    for i in range(number_of_generations): # number of generations
        gen_i = evolve(gen_i, train_inputs, train_targets)
        print(f"Gen{i + 1}: {sorted(evaluate_population(gen_i, train_inputs, train_targets), reverse = True)}")

        if i % 50 == 0 or i == number_of_generations - 1:
            print(f"Last Gen on Validate: {sorted(evaluate_population(gen_i, validate_inputs, validate_targets), reverse = True)}")


if __name__ == '__main__':
    main()
