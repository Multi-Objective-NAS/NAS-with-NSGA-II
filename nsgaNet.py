import constants
import functools
import numpy as np
import random
import matplotlib.pyplot as plt


def random_spec(population):
    ops = [constants.CONV3X3 for _ in range(7)]
    ops[0] = constants.INPUT
    ops[-1] = constants.OUTPUT
    while True:
        rand_binary_str = np.random.choice([0, 1], size=10)
        matrix = decode(rand_binary_str).tolist()
        spec = constants.api.ModelSpec(matrix=matrix, ops=ops)
        if possible_to_get_in(population, spec):
            return spec


def encode(matrix):
    binary_string = []

    for c in range(2, 6):
        for r in range(1, c):
            binary_string.append(matrix[r][c])

    return binary_string


def decode(binary_string):
    # initialize
    adjacency_mat = np.zeros((7, 7), dtype=int)
    idx = 0

    in_degree = np.zeros(7)
    out_degree = np.zeros(7)

    # fill adjacency matrix as binary string
    for c in range(2, 6):
        for r in range(1, c):
            adjacency_mat[r][c] = binary_string[idx]
            idx += 1
            in_degree[c] += 1
            out_degree[r] += 1

    for node in range(1, 6):
        if in_degree[node] == 0 and out_degree[node] != 0:
            adjacency_mat[0][node] = 1
        if in_degree[node] != 0 and out_degree[node] == 0:
            adjacency_mat[node][6] = 1

    return adjacency_mat


'''
Binary tournament operation.
Select two parent from randomly chosen 'tournament_size' of population.
'''


def parent_selection(population, tournament_size):
    population_size = len(population)
    selected = []
    for _ in range(2):
        pool = random.sample(population, tournament_size)
        sorted(pool, key=functools.cmp_to_key(crowded_comparison_operator))
        selected.append(pool[0]['spec'].original_matrix)
    return selected

'''
Crossover from two selected population members as parents.
Preserve common building blocks.
Maintain complexity.
'''


def crossover(parent1, parent2, crossover_prob):
    offspring = []
    if random.random() < crossover_prob:
        for p1, p2 in zip(parent1, parent2):
            if p1 != p2:
                offspring.append(random.randrange(0, 2))
            else:
                offspring.append(p1)
    else:
        offspring = parent1
    return offspring

'''
Bit flipping at most once
'''


def mutation(offspring, mutation_rate):
    if random.random() < mutation_rate:
        index = random.sample(range(len(offspring)), 1)[0]
        # bitwise
        offspring[index] = (1 + offspring[index]) % 2

    return offspring


def possible_to_get_in(pool, spec):
    if not constants.nasbench.is_valid(spec):
        return False
    for p in pool:
        if equal_model(p['spec'], spec):
            return False
    return True


def equal_model(present_spec, new_spec):
    if np.shape(present_spec.matrix)[0] != np.shape(new_spec.matrix)[0]:
        return False
    size = np.shape(present_spec.matrix)[0]
    return all([present_spec.matrix[row][col] == new_spec.matrix[row][col] for row in range(size) for col in range(row+1, size)])
    '''
    binary_str1 = encode(matrix1)
    binary_str2 = encode(matrix2)
    return all([binary_str1[idx] == binary_str2[idx] for idx in range(len(binary_str1))])
    '''


def generate_offspring(population,
                       generation_size,
                       tournament_size,
                       crossover_prob,
                       mutation_rate):
    # initialize
    offspring_population = []
    ops = [constants.CONV3X3 for _ in range(7)]
    ops[0] = constants.INPUT
    ops[-1] = constants.OUTPUT

    while len(offspring_population) < generation_size:
        # binary_tournament_selection
        parent1, parent2 = parent_selection(population, tournament_size)

        parent1 = encode(parent1)
        parent2 = encode(parent2)

        # crossover
        offspring = crossover(parent1, parent2, crossover_prob)

        # mutation
        offspring = mutation(offspring, mutation_rate)

        offspring_mat = decode(offspring).tolist()
        offspring_spec = constants.api.ModelSpec(matrix=offspring_mat, ops=ops)

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
            pass
        elif (elem1[obj] - elem2[obj]) * criteria > 0:
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
    if 'rank' in elem1.keys():
        if elem1['rank'] != elem2['rank']:
            return 1 if elem2['rank'] - elem1['rank'] > 0 else -1

    if elem1['dist'] == elem2['dist']:
        return 0
    else:
        return 1 if elem2['dist'] - elem1['dist'] > 0 else -1


def init_population(population, population_size):
    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    for _ in range(population_size):
        spec = random_spec(population)
        data = constants.nasbench.query(spec)
        elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': spec}
        population.append(elem)

    # Assign rank to do tournament selection.
    crowding_distance_assignment(population)
    fast_non_dominated_sort(population)


def visualize(population, turn, figure):
    color = ['#c62828', '#d81b60', '#8e24aa', '#3949ab', '#1e88e5',
             '#00897b', '#43a047', '#c0ca33', '#ffb300', '#ef6c00']
    subplot = figure.add_subplot(4, 5, turn)
    # print("[In this turn...]" + str(turn))
    for rank in range(1, len(color) + 1):
        accuracy = [person['acc'] for person in population if person['rank'] == rank]
        time = [person['time'] for person in population if person['rank'] == rank]
        # print(len(accuracy))
        subplot.scatter(time, accuracy, color=color[rank - 1], label='rank ' + str(rank))


'''
Main part of nsgaII
search_time=20
population size :  evolution pool size default 40.
generation size : 20
tournament size : 10
'''


def nsgaII(answer_size=40,
           search_time=2000,
           population_size=40,
           generation_size=10,
           tournament_size=5,
           crossover_prob=0.9,
           mutation_rate=0.1):
    constants.nasbench.reset_budget_counters()
    figure = plt.figure()
    population = []
    # Each element is one dictionary as { rank, validation accuracy , time, spec }
    init_population(population, population_size)
    # Initially Create random parent population

    # evolution
    for turn in range(1, search_time + 1):
        population += generate_offspring(population, generation_size, tournament_size, crossover_prob, mutation_rate)
        crowding_distance_assignment(population)
        fast_non_dominated_sort(population)
        sorted(population, key=functools.cmp_to_key(crowded_comparison_operator))
        population = population[:population_size]
        if turn % 100 == 0:
            visualize(population, turn//100, figure)

    '''
    # check
    for p in population:
        print("[person]")
        bin_str = encode(p['mat'])
        print(bin_str)
        print(p['acc'], p['time'])
    '''

    accuracy = [person['acc'] for person in population if person['rank'] == 1]
    time = [person['time'] for person in population if person['rank'] == 1]

    return accuracy, time
