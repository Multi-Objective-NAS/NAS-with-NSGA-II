import constants
import functools
import numpy as np
import random


def random_spec():
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(constants.ALLOWED_EDGES, size=(constants.NUM_VERTICES, constants.NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(constants.ALLOWED_OPS, size=constants.NUM_VERTICES).tolist()
        ops[0] = constants.INPUT
        ops[-1] = constants.OUTPUT
        spec = constants.api.ModelSpec(matrix=matrix, ops=ops)
        if constants.nasbench.is_valid(spec):
            return spec


def n_ary_to_m_ary(list_of_int, length, n, m):
    decimal = int(''.join(map(str, list_of_int)), base=n)

    # convert into m-ary list
    m_ary = []
    for _ in range(length):
        decimal, leftover = divmod(decimal, m)
        m_ary.append(leftover)

    return m_ary[::-1]


def binary_to_ternary(binary):
    return n_ary_to_m_ary(binary, length=5, n=2, m=3)


def ternary_to_binary(ternary):
    return n_ary_to_m_ary(ternary, length=8, n=3, m=2)


def encode(spec):
    #
    binary_code = []
    ternary_code = []

    for r in range(7):
        for c in range(r + 1, 7):
            binary_code.append(spec.matrix[r][c])

    for i in range(1, 6):
        for idx in range(3):
            if constants.ALLOWED_OPS[idx] == spec.ops[i]:
                ternary_code.append(idx)
                break

    binary_code.extend(ternary_to_binary(ternary_code))
    return binary_code


def decode(binary_code):
    matrix = np.zeros((7, 7))
    ops = [constants.INPUT]

    idx = 0
    for r in range(7):
        for c in range(r + 1, 7):
            matrix[r][c] = binary_code[idx]
            idx += 1

    binary_code = binary_code[idx:]
    ternary_code = binary_to_ternary(binary_code)

    for idx in range(5):
        ops.append(constants.ALLOWED_OPS[ternary_code[idx]])
    ops.append(constants.OUTPUT)

    model_spec = constants.api.ModelSpec(matrix, ops)
    return model_spec


def parent_selection(population, tournament_size=2):
    population_size = len(population)
    selected = []
    for _ in range(2):
        pool = random.sample(population, tournament_size)

        print("pool")
        for p in pool:
            print(p['rank'], p['dist'], end='    ')
        print()

        sorted(pool, key=functools.cmp_to_key(crowded_comparison_operator))
        selected.append(pool[0])
    print(selected)
    return selected


def crossover(parent1, parent2):
    index = random.sample(range(1, len(parent1)), 1)
    offspring = parent1[:index]
    offspring.extend(parent2[index:])
    return offspring


def mutation(offspring, mutation_rate):
    index = random.sample(range(1, len(offspring)), 1)
    # bitwise
    offspring[index] = (1 + offspring[index]) % 2
    return offspring


def generate_offspring(population,
                       crossover_prob,
                       tournament_size,
                       mutation_rate):
    population_size = len(population)
    size = 0
    offspring_population = []
    while size < population_size:
        # binary_tournament_selection
        parent1, parent2 = parent_selection(population, tournament_size)

        parent1 = encode(parent1)
        parent2 = encode(parent2)

        # crossover
        offspring = parent1
        if random.random() < crossover_prob:
            offspring = crossover(parent1, parent2)

        # mutation
        if random.random() < mutation_rate:
            offspring = mutation(offspring)

        offspring_spec = decode(offspring)

        if constants.nasbench.is_valid(offspring_spec):
            size += 1
            data = constants.nasbench.query(offspring_spec)
            elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': offspring_spec}
            offspring_population.append(elem)

    return offspring_population


def dominate_operator(elem1, elem2):
    dominate_count = [0, 0]
    idx = 0

    for obj in constants.OBJECTIVES:
        if elem1[obj] == elem2[obj]:
            pass
        elif elem1[obj] > elem2[obj]:
            if constants.OPT[idx] == 1:
                dominate_count[0] += 1
            elif constants.OPT[idx] == -1:
                dominate_count[1] += 1
        idx += 1

    if dominate_count[0] == 0 and dominate_count[1] > 0:
        return 1
    elif dominate_count[1] == 0 and dominate_count[0] > 0:
        return -1
    else:
        return 0


def fast_non_dominated_sort(population):
    S = []
    n = []
    sorted = {}

    pidx = 0
    for p in population:
        S.append(set())
        n.append(0)
        qidx = 0

        for q in population:
            judge = dominate_operator(p, q)
            if judge == -1:
                S[pidx].add(qidx)
            elif judge == 1:
                n[pidx] += 1
            qidx += 1

        if n[pidx] == 0:
            p['rank'] = 1
            if not 1 in sorted:
                sorted[1] = set()
            sorted[1].add(pidx)

        pidx += 1

    rank = 1
    while len(sorted[rank]) != 0:
        sorted[rank + 1] = set()

        for pidx in sorted[rank]:
            for qidx in S[pidx]:
                n[qidx] -= 1
                if n[qidx] == 0:
                    population[qidx]['rank'] = rank
                    sorted[rank + 1].add(qidx)
        rank += 1

    return sorted


def crowding_distance_assignment(population):
    # initialize
    for elem in population:
        elem['dist'] = 0

    for obj in constants.OBJECTIVES:
        population.sort(key=lambda e: e[obj])
        max_val = population[-1][obj]
        min_val = population[0][obj]

        start = 0
        end = len(population) - 1
        while start <= end and population[start][obj] == min_val:
            population[start]['dist'] = constants.INFINITE
            start += 1
        while start <= end and population[end][obj] == max_val:
            population[end]['dist'] = constants.INFINITE
            end -= 1

        pre_dist = min_val
        for i in range(start, end + 1):
            cur_dist = population[i]['dist']
            if cur_dist == constants.INFINITE:
                continue
            next_dist = population[i + 1][obj]
            cur_dist += (next_dist - pre_dist) / (max_val - min_val)
            population[i]['dist'] = cur_dist
            pre_dist = population[i][obj]

    print(population)


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
        spec = random_spec()
        data = constants.nasbench.query(spec)
        elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': spec}
        population.append(elem)

    # To do tournament selection.
    crowding_distance_assignment(population)
    fast_non_dominated_sort(population)

def nsgaII(answer_size=500,
           search_time=25000,
           population_size=500,
           crossover_prob=0.9,
           tournament_size=10,
           mutation_rate=0.03):
    constants.nasbench.reset_budget_counters()
    # 2 objectives: validation, time
    population = []  # element is { rank, validation accuracy , time, spec } dictionary
    init_population(population, population_size)

    # evolution
    # for _ in range(search_time):
    # population += generate_offspring(population, crossover_prob, tournament_size, mutation_rate)
    # fast_non_dominated_sort()
    # number = 0
