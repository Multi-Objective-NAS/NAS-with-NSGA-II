import constants
import nsgaNet
import numpy as np


def satisfy_condition(mat, op):
    length = len(op)
    if not all([op == constants.CONV3X3 for op in op[1:length - 1]]):
        return False

    return True


def pareto_front():
    answer_list = []
    total_data = []
    # element = {'acc': , 'time': }

    for hash_val in constants.nasbench.hash_iterator():
        fixed_stat, computed_stat = constants.nasbench.get_metrics_from_hash(hash_val)
        mat = fixed_stat['module_adjacency']
        op = fixed_stat['module_operations']
        if satisfy_condition(mat, op):
            model_spec = constants.api.ModelSpec(matrix=mat, ops=op)
            data = constants.nasbench.query(model_spec)
            new_elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'mat': model_spec.matrix}
            total_data.append(new_elem)

            append_flag = True
            idx = 0
            for elem in answer_list[:]:
                judge = nsgaNet.dominate_operator(elem, new_elem)
                if judge > 0:
                    answer_list.remove(elem)
                    idx -= 1
                elif judge < 0:
                    append_flag = False
                    break
            if append_flag:
                answer_list.append(new_elem)

    tot_accuracy = [elem['acc'] for elem in total_data]
    tot_time = [elem['time'] for elem in total_data]
    ans_accuracy = [elem['acc'] for elem in answer_list]
    ans_time = [elem['time'] for elem in answer_list]
    print("[total data set size]", len(total_data))

    return tot_accuracy, tot_time, ans_accuracy, ans_time
