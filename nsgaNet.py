import constants
import nasbench.lib.graph_util as gu
import functools
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

'''
Return spec
'''


def random_generation(population, population_size):
    times = []
    hvs = []
    random_search_space = random.sample(constants.whole_hash_list, population_size)

    for hash_val in random_search_space:
        fixed_stat, computed_stat = constants.nasbench.get_metrics_from_hash(hash_val)
        mat = fixed_stat['module_adjacency']
        op = fixed_stat['module_operations']
        spec = constants.api.ModelSpec(matrix=mat, ops=op)
        data = constants.nasbench.query(spec)
        new_elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': spec}
        population.append(new_elem)
        fast_non_dominated_sort(population)
        count_hypervolume(population, times=times, hvs=hvs)
    return times, hvs


'''
Return {matrix_code, operation_code}
'''


def encode(spec):
    matrix = spec.matrix
    ops = spec.ops

    # make 7x7 matrix and 7 operation
    size = np.shape(matrix)[0]
    adjacency_mat = np.zeros((7, 7), dtype=int)
    adjacency_mat[:size - 1, :size - 1] = np.copy(matrix[:size - 1, :size - 1])
    adjacency_mat[:size, 6] = np.copy(matrix[:size, size - 1])

    extended_ops = ops[:]
    extended_ops[-1:-1] = [constants.CONV3X3 for _ in range(7 - size)]
    '''
    size = np.shape(matrix)[0]
    matrix_code = []
    operation_code = [constants.ALLOWED_OPS.index(op) for op in ops[1:-1]]

    for c in range(size):
        for r in range(c):
            matrix_code.append(matrix[r][c])
    '''
    return {'mat': adjacency_mat, 'ops': extended_ops}


'''
Return spec
'''


def decode(cell):
    '''
    # initialize
    adjacency_mat = np.zeros((7, 7), dtype=int)
    idx = 0

    # fill adjacency matrix as binary string
    for c in range(7):
        for r in range(c):
            adjacency_mat[r][c] = matrix_code[idx]
            idx += 1

    ops = [constants.INPUT]
    ops += [constants.ALLOWED_OPS[idx] for idx in operation_code]
    ops.append(constants.OUTPUT)
    '''
    mat = cell['mat']
    ops = cell['ops']
    spec = constants.api.ModelSpec(matrix=mat.tolist(), ops=ops)
    return spec


'''
Binary tournament operation.
Select two parent from randomly chosen 'tournament_size' of population.
Return spec
'''


def parent_selection(population, tournament_size):
    population_size = len(population)
    selected = []
    for _ in range(2):
        pool = random.sample(population, tournament_size)
        pool = sorted(pool, key=functools.cmp_to_key(crowded_comparison_operator))
        selected.append(pool[0]['spec'])
    return selected


def graph_transform(parent1, parent2):
    mat1 = parent1['mat']
    label1 = parent1['ops']
    mat2 = parent2['mat']
    max_common = 0
    for perm in itertools.permutations(range(0, 7)):
        pmat1, plabel1 = gu.permute_graph(mat1, label1, perm)
        common = len(np.intersect1d(pmat1.flatten(), mat2.flatten()))
        if max_common < common:
            max_common = common
            parent1['mat'] = pmat1
            parent1['ops'] = plabel1


'''
Input, Output : (mat nparray, op list)
Crossover from two selected population members as parents.
Preserve common building blocks.
Maintain complexity.
'''


def crossover(parent1, parent2, crossover_prob):
    check = [False for _ in range(7)]
    check[6] = True
    offspring_mat = np.zeros((7, 7), dtype=int)
    if random.random() < crossover_prob:
        graph_transform(parent1, parent2)
        for idx in range(6, 0, -1):
            if check[idx] is False:
                continue
            p1 = np.asarray([node for node in range(0, idx) if parent1['mat'][node, idx] == 1], dtype=int)
            p2 = np.asarray([node for node in range(0, idx) if parent2['mat'][node, idx] == 1], dtype=int)
            # take over common part
            for prenode in np.intersect1d(p1, p2):
                offspring_mat[prenode, idx] = 1
                check[prenode] = True
            # randomly choose different part
            connected = np.union1d(p1, p2).tolist()
            choice_size = np.random.randint(len(connected)) + 1
            for prenode in random.sample(connected, choice_size):
                offspring_mat[prenode, idx] = 1
                check[prenode] = True
    offspring_ops = parent1['ops']
    for idx in range(7):
        if np.random.randint(2) == 0:
            offspring_ops[idx] = parent2['ops'][idx]
    return {'mat': offspring_mat, 'ops': offspring_ops}


'''
Bit flipping at most once
'''


def mutation(offspring, mutation_rate):
    if random.random() < mutation_rate:
        tonode = np.random.randint(6) + 1
        fromnode = np.random.randint(0, tonode)
        offspring['mat'][fromnode, tonode] = (offspring['mat'][fromnode, tonode] + 1) % 2
    if random.random() < mutation_rate:
        idx = np.random.randint(5) + 1
        op = np.random.randint(3)
        offspring['ops'][idx] = constants.ALLOWED_OPS[op]

    return offspring


def possible_to_get_in(pool, spec):
    if constants.nasbench.is_valid(spec) is False:
        return False
    else:
        for p in pool:
            if equal_model(p['spec'], spec):
                return False
        return True


def equal_model(present_spec, new_spec):
    present_mat = present_spec.matrix
    present_label = present_spec.ops
    new_mat = new_spec.matrix
    new_label = new_spec.ops
    if np.shape(present_mat) != np.shape(new_mat):
        return False
    else:
        return gu.is_isomorphic((present_mat.tolist(), present_label), (new_mat.tolist(), new_label))


def generate_offspring(population,
                       generation_size,
                       tournament_size,
                       crossover_prob,
                       mutation_rate):
    # initialize
    offspring_population = []

    while len(offspring_population) < generation_size:
        # binary_tournament_selection
        parent1, parent2 = parent_selection(population, tournament_size)

        # {'mat' : matrix, 'ops' : operation}
        parent1 = encode(parent1)
        parent2 = encode(parent2)

        # crossover
        offspring = crossover(parent1, parent2, crossover_prob)

        # mutation
        offspring = mutation(offspring, mutation_rate)

        offspring_spec = decode(offspring)

        if possible_to_get_in(offspring_population + population, offspring_spec):
            data = constants.nasbench.query(offspring_spec)
            elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': offspring_spec}
            offspring_population.append(elem)

    return offspring_population


'''
Returns which one Pareto dominated another.
'''


def dominate_operator(elem1, elem2):
    dominate_count = [0, 0]
    # Counts number of winning in each objectives.

    for obj, criteria in zip(constants.OBJECTIVES, constants.OPT):
        if elem1[obj] == elem2[obj]:
            continue
        elif ((elem1[obj] - elem2[obj]) * criteria) > 0.0:
            dominate_count[0] += 1
        else:
            dominate_count[1] += 1

    if dominate_count[0] == 0 and dominate_count[1] > 0:
        # elem2 dominates elem1
        return 1
    elif dominate_count[1] == 0 and dominate_count[0] > 0:
        # elem1 dominates elem2
        return -1
    else:
        return 0


'''
Assign rank as non-domination level.
'''


def fast_non_dominated_sort(population):
    S = []  # S[p] = set of solutions; the solution p dominates.
    n = []  # N[p] = domination count; the number of solutions which dominate p.
    sorted_by_rank = {}  # key = rank    value = set of indices.

    for p, p_idx in zip(population, range(len(population))):
        # initialize
        S.append(set())
        n.append(0)

        for q, q_idx in zip(population, range(len(population))):
            judge = dominate_operator(p, q)
            if judge == -1:
                # p dominates q
                S[p_idx].add(q_idx)
            elif judge == 1:
                # q dominates p
                n[p_idx] += 1

        if n[p_idx] == 0:
            p['rank'] = 1
            if not 1 in sorted_by_rank:
                sorted_by_rank[1] = set()
            sorted_by_rank[1].add(p_idx)

    pre_rank = 1
    next_rank = 2
    while len(sorted_by_rank[pre_rank]) != 0:
        sorted_by_rank[next_rank] = set()

        for p_idx in sorted_by_rank[pre_rank]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    population[q_idx]['rank'] = next_rank
                    sorted_by_rank[next_rank].add(q_idx)
        pre_rank = next_rank
        next_rank += 1


'''
Assign crowding distance as density estimation.
'''


def crowding_distance_assignment(population):
    # initialize
    for elem in population:
        elem['dist'] = 0

    # Calculate the sum of individual distance values
    # corresponding to each objective.
    for obj in constants.OBJECTIVES:
        population.sort(key=lambda e: e[obj])
        max_val = population[-1][obj]
        min_val = population[0][obj]
        population[0]['dist'] = population[-1]['dist'] = constants.INFINITE

        for i in range(1, len(population) - 1):
            cur_dist = population[i]['dist']
            if cur_dist == constants.INFINITE:
                continue
            cur_dist += (population[i + 1][obj] - population[i - 1][obj]) / (max_val - min_val)
            population[i]['dist'] = cur_dist


def crowded_comparison_operator(elem1, elem2):
    # elem is dictionary{'acc', 'time', 'spec', 'rank'}
    # return -1: elem1 is optimal / 1: elem2 is optimal.
    if elem1['rank'] != elem2['rank']:
        return -1 if (elem2['rank'] - elem1['rank']) > 0 else 1
    else:
        if elem1['dist'] == elem2['dist']:
            return 0
        else:
            return -1 if (elem2['dist'] - elem1['dist']) < 0 else 1


def init_population(population, population_size):
    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    random_generation(population, population_size)

    # Assign rank to do tournament selection.
    crowding_distance_assignment(population)
    # fast_non_dominated_sort(population)


def visualize(population, turn, ax):
    # red, orange, yellow, light green, green
    # light blue, blue, violet, brown
    color = ['#c62828', '#ef6c00', '#fdd835', '#7cb342', '#004d40',
             '#039be5', '#1a237e', '#7b1fa2', '#5d4037']
    markers = ['o', 'x', '*', '^', '>', '<']
    accuracy_list = [elem['acc'] for elem in population if elem['rank'] == 1]
    time_list = [elem['time'] for elem in population if elem['rank'] == 1]
    ax.scatter(time_list, accuracy_list, color=color[turn], marker=markers[turn], label='NSGA_' + str(turn))


def count_hypervolume(population, times, hvs):
    time_spent, _ = constants.nasbench.get_budget_counters()

    pareto_front = [elem for elem in population if elem['rank'] == 1]
    pareto_front.sort(key=lambda e: e['time'])
    HV = 0.0
    for idx in range(len(pareto_front) - 1):
        HV += (pareto_front[idx + 1]['time'] - pareto_front[idx]['time']) * pareto_front[idx]['acc']
    HV += (constants.time_inf - pareto_front[-1]['time']) * pareto_front[-1]['acc']
    times.append(time_spent)
    hvs.append(HV)


'''
Main part of nsgaII
search_time=20
population size :  evolution pool size default 40.
generation size : 20
tournament size : 10
'''


def generate_nsgaII_data(axes,
                         search_time=100,
                         population_size=40,
                         generation_size=2,
                         tournament_size=3,
                         crossover_prob=0.9,
                         mutation_rate=0.2):
    constants.nasbench.reset_budget_counters()
    population = []
    search_space = []
    # show image 5 times
    period = search_time // 5
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Accuracy')
    times = []
    hvs = []
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('HV')
    # Each element is one dictionary as { rank, validation accuracy , time, spec }
    init_population(population, population_size)
    # Initially Create random parent population

    # evolution
    for turn in range(1, search_time + 1):
        offspring = generate_offspring(population, generation_size, tournament_size, crossover_prob, mutation_rate)
        population += offspring
        search_space += offspring
        crowding_distance_assignment(population)
        fast_non_dominated_sort(population)
        population = sorted(population, key=functools.cmp_to_key(crowded_comparison_operator))
        population = population[:population_size]
        if turn % period == 0:
            visualize(population, turn // period, axes[0])
        count_hypervolume(population, times=times, hvs=hvs)

    # visualize_hypervolume
    axes[1].plot(times, hvs, label='NSGA', color='blue')

    return population
