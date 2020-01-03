import constants
import nsgaNet


def random_pareto_front(size):
    random_list = []
    random_search_space = []
    # element = {'acc': , 'time': }

    while len(random_list) <= size:
        spec = nsgaNet.random_spec(random_list)
        data = constants.nasbench.query(spec)
        new_elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': spec}

        if nsgaNet.possible_to_get_in(random_list, spec):
            random_list.append(new_elem)

    nsgaNet.crowding_distance_assignment(random_list)
    nsgaNet.fast_non_dominated_sort(random_list)

    pareto_accuracy = [elem['acc'] for elem in random_list if elem['rank'] == 1]
    pareto_time = [elem['time'] for elem in random_list if elem['rank'] == 1]
    ss_accuracy = [elem['acc'] for elem in random_list]
    ss_time = [elem['time'] for elem in random_list]

    return pareto_accuracy, pareto_time, ss_accuracy, ss_time
