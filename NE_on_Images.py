import numpy
import torch
import networks as nws
import random as rnd
import copy
import save_util as saveu
import image_util as imageu
from statistics import mean

def generate_inputs():
    train_set = []
    validate_set = []

    # get input vectors from paths
    for i in range(975, 990):
        train_set.append(imageu.png2torchtensor(f'D:/workFolder/NeuroEvolution/thumbnails/00{i}.png'))

    for i in range(10, 20):
        validate_set.append(imageu.png2torchtensor(f'D:/workFolder/NeuroEvolution/thumbnails/000{i}.png'))

    return train_set, validate_set

def softmax(rgb_values):
    return (numpy.exp(rgb_values) / numpy.sum(numpy.exp(rgb_values), axis=0)).tolist()

# Returns a list of tuples with 3 entries each
# These values will have to get softmaxed to represent percentage values of the
# grayscale
# TODO: change name
def result_to_list(result, fast):
    tuple_list = []
    result_list = torch.squeeze(result)
    result_list = torch.flatten(result_list, 1)
    result_list = result_list.tolist()
    r, g, b = result_list[0], result_list[1], result_list[2]
    for red, green, blue in zip(r, g, b):
        tuple_list.append(softmax([red, green, blue]))

    if not fast:
        print(r)
        print(g)
        print(b)
        print(tuple_list)

    return tuple_list if not fast else b

# populates a population with a network type
def populate(population_size):
    population = []
    for i in range(population_size):
        population.append(nws.ImageAutoencoderTwo())
    return population

# creates a new population
class nnPopulation:

    def __init__(self, population_size):
        self.population_size = population_size
        self.population = populate(population_size)


def fitness(network, inputs):
    # for now: maximize the blue influence
    # TODO: write this
    score = 0
    for input in inputs:
        result = network.forward(input)
        result = result_to_list(result, True)
        for value in result:
            score += value
        score /= 128 * 128
    score /= len(inputs)

    return score


def evaluate_population(population, inputs):
    results = []
    for network in population:
        results.append(fitness(network, inputs))
    return results


def getNextParents(nets_and_results, keep, type='roulette_wheel'):
    # TODO: Add other types of selecting functions
    # e.g. Elitism, Tournament Selection, SUS

    # net_list should probably be a net and result list, since we need results too

    # Methode for roulette wheel: fitness ranges from 0 to 0.99
    # we take this score * 100 and turn it into an into
    # each individual gets slices equal to that number
    # then randomly select 'keep' many

    parent_nets = []
    net_list = [i[0] for i in nets_and_results]
    # fitness_scores is sorted, since we get a sorted list as input
    fitness_scores = [i[1] for i in nets_and_results]

    if type == 'roulette_wheel':
        roulette_wheel = []
        list_counter = 0

        for score in fitness_scores:
            slices = int(score * 100)
            for i in range(slices):
                roulette_wheel.append(list_counter)
            list_counter = list_counter + 1

        print(roulette_wheel)

        rnd.shuffle(roulette_wheel) # in place shuffle
        for i in range(keep):
            winner = rnd.randrange(len(roulette_wheel))
            parent_nets.append(net_list[roulette_wheel[winner]])

    return parent_nets

# TODO: make getNextParents more general and give it type as argument
# no if else needed afterwards
def evolve(population, train_inputs, mutation_rate=0.07, keep=10, type='top'):
    """ First we apply the survival of the fittest principle """
    nets_and_results = list(zip(population, evaluate_population(population, train_inputs)))
    # sort in place
    nets_and_results.sort(key=lambda x: x[1], reverse = True)

    if type == 'top':
        # since we sorted in place, new_nets is also sorted
        new_nets = [i[0] for i in nets_and_results]

        # get the size of the population to later repop it to the same size
        size = len(new_nets)

        # delete everything but the top 'keep' individuals
        # TODO: Maybe use a "getFittest" function here to select the parents
        del new_nets[keep:]

    elif type == 'roulette':

        new_nets = getNextParents(nets_and_results, keep=keep)
        size = len(nets_and_results)

    # fill the population back up to 'size'
    filler = []

    for i in range(len(new_nets)): # before 'keep'
        filler.append(copy.deepcopy(new_nets[i]))

    while len(new_nets) < size:
        new_nets.extend(copy.deepcopy(filler))

    """ After that we mutate the fittest individuals """
    for net in new_nets:

        layers = net.get_layers()

        for layer in layers:

            delta = numpy.random.randn(*layer.weight.data.size()).astype(numpy.float32) * numpy.array(mutation_rate).astype(numpy.float32)
            delta_tensor = torch.from_numpy(delta)
            layer.weight.data = layer.weight.data.add(delta_tensor)
    # Stop V2

    return new_nets


def main():
    number_of_generations = 30
    size_of_population = 6
    mutation_rate = 0.05
    keep = 2
    results = []

    test_pop = nnPopulation(size_of_population)

    # training dataset, validation dataset
    train_inputs, validate_inputs = generate_inputs()

    gen_i = test_pop.population # = gen_0 aka initial population

    for i in range(number_of_generations + 1): # number of generations
        if i == 0:
            # initial generation only has to get evaluated
            gen_i_eval = sorted(evaluate_population(gen_i, train_inputs), reverse = True)
            print(f"Gen {i}: {gen_i_eval}")
        else:
            # we need to evolve first if it's not the initial generation
            # the evolve turns the gen into the gen + 1 i.e. the new gen
            gen_i = evolve(gen_i, train_inputs, mutation_rate=mutation_rate, keep=keep)

            gen_i_eval = sorted(evaluate_population(gen_i, train_inputs), reverse = True)
            print(f"Gen {i}: {gen_i_eval}")

        gen_i_eval_result = [i] # Generation number
        gen_i_eval_result.extend(gen_i_eval) # Fitness Scores (best to worst)
        gen_i_eval_result.append(mean(gen_i_eval)) # Average
        results.append(gen_i_eval_result)

        if i == number_of_generations: # gens to test on vali
            gen_i_vali = sorted(evaluate_population(gen_i, validate_inputs), reverse = True)
            print(f"Validation Gen {i}: {gen_i_vali}")

            # TODO: add back later: gen_i_vali_result = [f'Validation Gen {i}']
            gen_i_vali_result = [-1]
            gen_i_vali_result.extend(gen_i_vali)
            gen_i_vali_result.append(mean(gen_i_vali))
            results.append(gen_i_vali_result)
            # ..._result is the string ready to add, the non ..._result version is the actual result list

            # Shows the resulting image / vector in image shape
            imageu.show_result_image(result_to_list(gen_i[0].forward(validate_inputs[0]), False), 'D:/workFolder/NeuroEvolution/thumbnails/00010.png')

    # save the results
    saveu.save_results(results)

    # TODO: Save more information into the result
    # e.g. Average, Information on mutation rate, pop size, keep, number of generations

if __name__ == '__main__':
    main()
