import constants
import nsgaNet
import numpy as np


def satisfy_condition(mat, op):
    length = len(op)
    if not all([op == constants.CONV3X3 for op in op[1:length - 1]]):
        return False
    in_degree = np.zeros(7)
    out_degree = np.zeros(7)

    # fill adjacency matrix as binary string
    for c in range(2, length - 1):
        for r in range(1, c):
            in_degree[c] += 1
            out_degree[r] += 1

    if mat[0][length - 1] == 1:
        return False
    for node in range(1, length - 1):
        if in_degree[node] != 0 and mat[0][node] == 1:
            return False
        if out_degree[node] != 0 and mat[node][length - 1] == 1:
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
            while idx < len(answer_list):
                elem = answer_list[idx]
                judge = nsgaNet.dominate_operator(elem, new_elem)
                if judge > 0:
                    answer_list.remove(elem)
                    idx -= 1
                elif judge < 0:
                    append_flag = False
                    break
                idx += 1
            if append_flag:
                answer_list.append(new_elem)

    tot_accuracy = [elem['acc'] for elem in total_data]
    tot_time = [elem['time'] for elem in total_data]
    ans_accuracy = [elem['acc'] for elem in answer_list]
    ans_time = [elem['time'] for elem in answer_list]

    return tot_accuracy, tot_time, ans_accuracy, ans_time
