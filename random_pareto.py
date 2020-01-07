import constants
import nsgaNet
import random


def random_pareto_front(size):
    random_list = []
    # element = {'acc': , 'time': }
    nsgaNet.random_generation(random_list, size)

    nsgaNet.crowding_distance_assignment(random_list)
    nsgaNet.fast_non_dominated_sort(random_list)

    pareto_accuracy = [elem['acc'] for elem in random_list if elem['rank'] == 1]
    pareto_time = [elem['time'] for elem in random_list if elem['rank'] == 1]
    ss_accuracy = [elem['acc'] for elem in random_list]
    ss_time = [elem['time'] for elem in random_list]

    return pareto_accuracy, pareto_time, ss_accuracy, ss_time
