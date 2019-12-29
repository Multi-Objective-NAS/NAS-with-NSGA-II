import constants
import nsgaNet

def pareto_front():
    answer_list = []
    # element = {'acc': , 'time': }

    for hash_val in constants.api.hash_iterator():
        fixed_stat, computed_stat = constants.api.get_metrics_from_hash(hash_val)
        if all([op == constants.CONV3X3 for op in fixed_stat['module_operations'][1:6]]):
            model_spec = constants.api.ModelSpec(matrix = fixed_stat['module_adjacency'], ops = fixed_stat['module_operations'])
            data = constants.api.query(model_spec)
            new_elem = {'acc': data['validation_accuracy'], 'time': data['training_time']}
            if len(answer_list) == 0:
                answer_list.append(new_elem)
                continue
            for elem in answer_list:
                if nsgaNet.dominate_operator(elem, new_elem) < 0:
                    answer_list.remove(elem)
                    answer_list.append(new_elem)
                    break

    accuracy = [elem['acc'] for elem in answer_list]
    time = [elem['time'] for elem in answer_list]

    return accuracy, time
