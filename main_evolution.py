import numpy
import torch
import torch.nn as nn
import random
import os
import copy
import save_util as saveu
import image_util as imageu
import nima_util as nimau
import squeezenet_util as squeezeu
from statistics import mean

import time
import concurrent.futures

dir_path = os.path.dirname(os.path.realpath(__file__))

def generate_inputs():
    """
    Generates the inputs (train and validation set) for the project pipelin out
    of the given image paths.

    Returns:
        Returns two lists containing the given images as pytorch vectors ready
        to use as inputs in squeezenet1_1.
        train_set: Training dataset.
        validate_set: Validation dataset.
    """
    train_set = []
    validate_set = []

    train_paths = []
    validate_paths = []

    for i in range(1001, 5001):
        if (i != 2015) and (i != 2746):
            train_paths.append(dir_path + f'\\dataset\\dataset({i}).jpg')
    for i in range(1, 1001):
        validate_paths.append(dir_path + f'\\dataset\\dataset({i}).jpg')

    with concurrent.futures.ThreadPoolExecutor() as executer:
        results1 = executer.map(imageu.to_squeezenet_vector, train_paths)
        train_set = [vector for vector in results1]

        results2 = executer.map(imageu.to_squeezenet_vector, validate_paths)
        validate_set = [vector for vector in results2]

    return train_set, validate_set

def result_to_list(result):
    return result.tolist()[0]

def populate(population_size):
    """
    Generates a population_size big population for the evolutionary algorithm.

    Args:
        population_size: Size of the created population.

    Returns:
        Returns the population as a list of pytorch Conv2D layers.
    """
    population = []
    for i in range(population_size):
        population.append(nn.Conv2d(512, 3, kernel_size=1))

    # initialise own weight values
    for indiv in population:
        initial_value = numpy.random.randn(*indiv.weight.data.size()).astype(numpy.float32)
        initial_value_tensor = torch.from_numpy(initial_value)
        indiv.weight.data = initial_value_tensor
    return population

# this function takes the longest (the more pictures the longer)
def fitness(finalconv, inputs):
    """
    This function calculates the fitness to a given individual of the current
    population in regard to the given inputs. It uses NIMA to calculate the
    final fitness score.

    Args:
        finalconv: Individual of the current generation. A pytorch Conv2D layer
            of size fit to the squeezenet1_1 final_conv layer.
        inputs: List of pytorch vectors individually used as inputs for
            squeezenet1_1.

    Return:
        Returns the fitness score calculated by the NIMA model (range 1 to 10).
    """
    score = 0
    rgbs = []
    scores = []

    rgbs = squeezeu.run_and_get_values(finalconv, inputs)
    # nima_vectors is a list of all input images in grayscale in nima input size
    nima_vectors = []
    for input, rgb in zip(inputs, rgbs):
        nima_vectors.append(imageu.network_and_rgb_to_nima_vector(input, rgb))

    with concurrent.futures.ProcessPoolExecutor() as executer:
        results1 = executer.map(nimau.evaluate_single_mean, nima_vectors)
        scores = [result for result in results1]
    score = mean(scores)

    return score

def evaluate_population(population, inputs):
    """
    Evaluates the current population.

    Args:
        population: List of individuals.
        inputs: List of pytorch vectors individually used as inputs for
            squeezenet1_1.

    Return:
        Returns a list of tupels. Each tupel contains the individual at index 0
        and the fitness score at index 1.
    """

    results = []
    for individual in population:
        results.append((individual, fitness(individual, inputs)))

    return results

def getNextParents(finalconvs_and_results, keep, type):
    """
    This function chooses the parents for the next generation according the
    given selection method.

    Args:
        finalconvs_and_results: List of individuals and respective fitness
            scores.
        keep: Number of individuals to be kept for the next generation.
        type: Selection method.

    Returns:
        Returns the parents for the next generation as a list.
    """
    # TODO:Do we want to use Elitism?

    parent_nets = []
    finalconvs_list = [i[0] for i in finalconvs_and_results]
    # fitness_scores is sorted, since we get a sorted list as input
    fitness_scores = [i[1] for i in finalconvs_and_results]

    if type == 'roulette':
        #TODO: Might want to turn the weights in as fractions with sum = 1
        parent_nets.extend(random.choices(finalconvs_list, fitness_scores, k=keep))

    elif type == 'sus':
        sum_fitness = 0
        for score in fitness_scores:
            sum_fitness += score
        first_pointer = numpy.random.uniform(0, sum_fitness)
        pointer_dist = sum_fitness / keep
        winners = [(first_pointer + i * pointer_dist) % sum_fitness for i in range(keep)]

        for winner in winners:
            temp_scores = copy.deepcopy(fitness_scores)
            temp_scores = numpy.cumsum(temp_scores).tolist()
            for score in temp_scores:
                if score > winner:
                    parent_nets.append(finalconvs_list[temp_scores.index(score)])
                    break

    elif type == 'rank':
        ranks = [x + 1 for x in reversed(range(len(finalconvs_list)))]
        parent_nets.extend(random.choices(finalconvs_list, ranks, k=keep))

    elif type == 'truncation':
        parent_nets = [i[0] for i in finalconvs_and_results[:keep]]

    # for all tournament selection variants keep should be 2 or greater!
    elif type == 'tournament-replacement': # with replacement
        group_size = 5
        participant_ids = [x for x in range(len(fitness_scores))]
        participants = [(id, score) for id, score in zip(participant_ids, fitness_scores)]
        while len(parent_nets) < keep:
            random.shuffle(participants)
            competing = random.choices(participants, k=group_size)
            competing.sort(key=lambda x: x[1], reverse=True)
            parent_nets.append(finalconvs_list[competing[0][0]])

    elif type == 'tournament': #without replacement per group
        group_size = int(len(fitness_scores) / keep)
        participant_ids = [x for x in range(len(fitness_scores))]
        participants = [(id, score) for id, score in zip(participant_ids, fitness_scores)]
        while len(parent_nets) < keep:
            random.shuffle(participants)
            competing = participants[:group_size]
            competing.sort(key=lambda x: x[1], reverse=True)
            parent_nets.append(finalconvs_list[competing[0][0]])

    else:
        raise Exception("No method of parent selection was given.")
    return parent_nets

def evolve(population_and_results, uses_elitism, mutation_rate=0.07, keep=10, type='truncation'):
    """
    This function evolves a given population to a new one.

    Args:
        population_and_results: List of tupels each containing an individual and
            its result.
        mutation_rate: Mutation base value.
        keep: Individuals to keep in the process for the next generation.
        type: Selection method.

    Return:
        Returns a new generation of pytorch Conv2D layers as a list.
    """
    """ First we apply the survival of the fittest principle """
    finalconvs_and_results = population_and_results
    # sort after fitness scores
    finalconvs_and_results.sort(key=lambda x: x[1], reverse = True)

    best_finalconv = finalconvs_and_results[0][0]

    size = len(finalconvs_and_results)

    new_finalconvs = getNextParents(finalconvs_and_results, keep=keep, type=type)

    # fill the population back up to the old/original size
    filler = []

    for i in range(len(new_finalconvs)):
        filler.append(copy.deepcopy(new_finalconvs[i]))

    while len(new_finalconvs) < size:
        new_finalconvs.extend(copy.deepcopy(filler))

    if uses_elitism:
        new_finalconvs[-1] = best_finalconv

    """ After that we mutate the fittest individuals/new population """
    for convlayer in new_finalconvs:
        delta = numpy.random.randn(*convlayer.weight.data.size()).astype(numpy.float32) * numpy.array(mutation_rate).astype(numpy.float32)
        delta_tensor = torch.from_numpy(delta)
        convlayer.weight.data = convlayer.weight.data.add(delta_tensor)

    return new_finalconvs

def main():
    """
    This is the main function of the whole NeuroEvolution project.

    number_of_generations: number of generations this will run for
    size_of_population: number of individuals per population
    mutation_rate:
    keep: number of individuals kept as parents for the next generation
    parent_selection_type: way of selecting parents

    It is needed for keep to be a divider of size_of_population!
    """
    load_generation = 0

    number_of_generations = 1000
    size_of_population = 100
    mutation_rate = 0.005
    keep = 5
    parent_selection_type = 'tournament-replacement'
    elitism = False
    results = []

    total_generations = number_of_generations + load_generation
    save_generation = total_generations

    population = populate(size_of_population)

    # create training dataset, validation dataset
    train_inputs, validate_inputs = generate_inputs()

    gen_i = population # = gen_0 aka initial population
    if load_generation > 0:
        gen_i = saveu.load_generation(load_generation, size_of_population)
    # following two could be in loop
    gen_i_eval = []
    gen_i_fitness_list = []

    for i in range(load_generation, total_generations + 1): # number of generations + initial pop

        if i == load_generation:
            # initial generation only has to get evaluated
            gen_i_eval = evaluate_population(gen_i, train_inputs)
            gen_i_fitness_list = sorted([x[1] for x in gen_i_eval], reverse=True)
            print(f"Gen {i}: {gen_i_fitness_list}")
        else:
            # we need to evolve first if it's not the initial generation
            # the evolve turns the gen into the gen + 1 i.e. the new gen
            gen_i = evolve(gen_i_eval, uses_elitism=elitism, mutation_rate=mutation_rate, keep=keep, type=parent_selection_type)

            gen_i_eval = evaluate_population(gen_i, train_inputs)
            gen_i_fitness_list = sorted([x[1] for x in gen_i_eval], reverse=True)
            print(f"Gen {i}: {gen_i_fitness_list}")

        gen_i_eval_result = [i] # Generation number
        gen_i_eval_result.extend(gen_i_fitness_list) # Fitness Scores (best to worst)
        gen_i_eval_result.append(mean(gen_i_fitness_list)) # Average
        results.append(gen_i_eval_result)

        # save each generation
        if i != load_generation:
            gen_i_eval.sort(key=lambda x: x[1], reverse=True)
            saveu.save_generation([x[0] for x in gen_i_eval], i)

        if i == total_generations or total_generations % 50 == 0: # gens to test on vali
            gen_i_vali_eval = evaluate_population(gen_i, validate_inputs)
            gen_i_vali_fitness_list = sorted([x[1] for x in gen_i_vali_eval], reverse=True)
            print(f"Validation Gen {i}: {gen_i_vali_fitness_list}")

            # TODO: add back later: gen_i_vali_result = [f'Validation Gen {i}']
            gen_i_vali_result = [-1 * i]
            gen_i_vali_result.extend(gen_i_vali_fitness_list)
            gen_i_vali_result.append(mean(gen_i_vali_fitness_list))
            results.append(gen_i_vali_result)
            # ..._result is the string ready to add, the non ..._result version is the actual result list


    # save the results
    """ for when you want to save stuff #saveu.save_results(results) """
    saveu.save_results(results)

if __name__ == '__main__':
    main()
