import nsgaNet
import constants


def random_pareto_front():
    random_list = []
    # element = {'acc': , 'time': }

    while len(random_list) <= 30:
        spec = nsgaNet.random_spec(random_list)
        data = constants.nasbench.query(spec)
        new_elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': spec}

        if nsgaNet.possible_to_get_in(random_list, spec):
            random_list.append(new_elem)

    nsgaNet.crowding_distance_assignment(random_list)
    nsgaNet.fast_non_dominated_sort(random_list)

    accuracy = [elem['acc'] for elem in random_list if elem['rank'] == 1]
    time = [elem['time'] for elem in random_list if elem['rank'] == 1]

    return accuracy, time


def random_data():
    random_list = []
    # element = {'acc': , 'time': }

    while len(random_list) <= 300:
        spec = nsgaNet.random_spec(random_list)
        data = constants.nasbench.query(spec)
        new_elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': spec}

        if nsgaNet.possible_to_get_in(random_list, spec):
            random_list.append(new_elem)

    accuracy = [elem['acc'] for elem in random_list]
    time = [elem['time'] for elem in random_list]

    return accuracy, time
