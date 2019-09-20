import numpy
import torch
import torch.nn as nn
import copy
import save_util as saveu
import image_util as imageu
import nima_util as nimau
import squeezenet_util as squeezeu
from statistics import mean
from datetime import datetime


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

    # paths to the images in the string
    for i in range(5, 22):
        train_set.append(imageu.to_squeezenet_vector(f'image_path'))

    for i in range(1, 5):
        validate_set.append(imageu.to_squeezenet_vector(f'image_path'))

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

    # scores is all the mean values returned from nima
    scores = nimau.evaluate_images(nima_vectors)[0]
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
    # TODO: Add other types of selecting functions
    # e.g. Elitism, Tournament Selection, SUS

    parent_nets = []
    finalconvs_list = [i[0] for i in finalconvs_and_results]
    # fitness_scores is sorted, since we get a sorted list as input
    fitness_scores = [i[1] for i in finalconvs_and_results]

    # TODO: What kind of distribution is needed?
    if type == 'roulette_wheel':
        sum_fitness = 0
        for score in fitness_scores:
            sum_fitness += score
        cumulative_fitness_scores = numpy.cumsum(fitness_scores)#.tolist()
        for i in range(keep):
            winner = sum_fitness * numpy.random.uniform(0, 1)
            parent_nets.append(finalconvs_list[numpy.argmax(cumulative_fitness_scores > winner)])

    elif type == 'rank':
        cumulative_rank = numpy.cumsum(list(range(1, len(finalconvs_list) + 1)))
        pieces_per_rank = cumulative_rank[::-1]
        cumulative_rank = numpy.cumsum(cumulative_rank[::-1])
        for i in range(keep):
            winner = numpy.random.randint(0, numpy.sum(pieces_per_rank))
            parent_nets.append(finalconvs_list[numpy.argmax(cumulative_rank > winner)])

    elif type == 'truncation':
        parent_nets = [i[0] for i in finalconvs_and_results]
        del parent_nets[keep:]

    else:
        raise Exception("No method of parent selection was given.")
    return parent_nets

def evolve(population_and_results, mutation_rate=0.07, keep=10, type='truncation'):
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

    size = len(finalconvs_and_results)

    new_finalconvs = getNextParents(finalconvs_and_results, keep=keep, type=type)

    # fill the population back up to the old/original size
    filler = []

    for i in range(len(new_finalconvs)):
        filler.append(copy.deepcopy(new_finalconvs[i]))

    while len(new_finalconvs) < size:
        new_finalconvs.extend(copy.deepcopy(filler))

    """ After that we mutate the fittest individuals/new population """
    for convlayer in new_finalconvs:
        delta = numpy.random.randn(*convlayer.weight.data.size()).astype(numpy.float32) * numpy.array(mutation_rate).astype(numpy.float32)
        delta_tensor = torch.from_numpy(delta)
        convlayer.weight.data = convlayer.weight.data.add(delta_tensor)

    return new_finalconvs

def main():
    # TODO: keep has to be a divider of size_of_population! Rework this (probably)
    """
    This is the main function of the whole NeuroEvolution project.

    number_of_generations: number of generations this will run for
    size_of_population: number of individuals per population
    mutation_rate:
    keep: number of individuals kept as parents for the next generation
    parent_selection_type: way of selecting parents
    """
    number_of_generations = 10
    size_of_population = 4
    mutation_rate = 0.05
    keep = 2
    parent_selection_type = 'truncation'
    results = []

    population = populate(size_of_population)

    # TODO: if we can get this out of here and just load a csv we could save time
    # create training dataset, validation dataset
    train_inputs, validate_inputs = generate_inputs()

    gen_i = population # = gen_0 aka initial population
    # following two could be in loop
    gen_i_eval = []
    gen_i_fitness_list = []

    for i in range(number_of_generations + 1): # number of generations + initial pop

        if i == 0:
            # initial generation only has to get evaluated
            gen_i_eval = evaluate_population(gen_i, train_inputs)
            gen_i_fitness_list = sorted([x[1] for x in gen_i_eval], reverse=True)
            print(f"Gen {i}: {gen_i_fitness_list}")
        else:
            # we need to evolve first if it's not the initial generation
            # the evolve turns the gen into the gen + 1 i.e. the new gen
            gen_i = evolve(gen_i_eval, mutation_rate=mutation_rate, keep=keep, type=parent_selection_type)

            gen_i_eval = evaluate_population(gen_i, train_inputs)
            gen_i_fitness_list = sorted([x[1] for x in gen_i_eval], reverse=True)
            print(f"Gen {i}: {gen_i_fitness_list}")

        gen_i_eval_result = [i] # Generation number
        gen_i_eval_result.extend(gen_i_fitness_list) # Fitness Scores (best to worst)
        gen_i_eval_result.append(mean(gen_i_fitness_list)) # Average
        results.append(gen_i_eval_result)

        if i == number_of_generations: # gens to test on vali
            gen_i_vali_eval = evaluate_population(gen_i, validate_inputs)
            gen_i_vali_fitness_list = sorted([x[1] for x in gen_i_vali_eval], reverse=True)
            print(f"Validation Gen {i}: {gen_i_vali_fitness_list}")

            # TODO: add back later: gen_i_vali_result = [f'Validation Gen {i}']
            gen_i_vali_result = [-1]
            gen_i_vali_result.extend(gen_i_vali_fitness_list)
            gen_i_vali_result.append(mean(gen_i_vali_fitness_list))
            results.append(gen_i_vali_result)
            # ..._result is the string ready to add, the non ..._result version is the actual result list

            # save / show result image(s)

    # save the results
    """ for when you want to save stuff #saveu.save_results(results) """
    # TODO: Save more information into the result
    # e.g. Average, Information on mutation rate, pop size, keep, number of generations

if __name__ == '__main__':
    main()
