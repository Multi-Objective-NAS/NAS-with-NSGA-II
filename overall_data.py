import constants
import nsgaNet
import numpy as np


def satisfy_condition(mat, op):
    '''length = len(op)
    if not all([op == constants.CONV3X3 for op in op[1:length - 1]]):
        return False
    '''
    return True


def generate_all_data():
    total_data = []
    # element = {'acc': , 'time': }

    for hash_val in constants.nasbench.hash_iterator():
        fixed_stat, computed_stat = constants.nasbench.get_metrics_from_hash(hash_val)
        mat = fixed_stat['module_adjacency']
        op = fixed_stat['module_operations']
        if satisfy_condition(mat, op):
            model_spec = constants.api.ModelSpec(matrix=mat, ops=op)
            data = constants.nasbench.query(model_spec)
            new_elem = {'acc': data['validation_accuracy'], 'time': data['training_time']}
            total_data.append(new_elem)
    #tot_accuracy = [elem['acc'] for elem in total_data]
    #tot_time = [elem['time'] for elem in total_data]
    print("[total data set size]", len(total_data))

    return total_data
