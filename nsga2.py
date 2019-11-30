import setup as su

def random_spec():
  """Returns a random valid spec."""
  while True:
    matrix = su.np.random.choice(su.ALLOWED_EDGES, size=(su.NUM_VERTICES, su.NUM_VERTICES))
    matrix = su.np.triu(matrix, 1)
    ops = su.np.random.choice(su.ALLOWED_OPS, size=(su.NUM_VERTICES)).tolist()
    ops[0] = su.INPUT
    ops[-1] = su.OUTPUT
    spec = su.api.ModelSpec(matrix=matrix, ops=ops)
    if su.nasbench.is_valid(spec):
      return spec

def binary_to_ternary(binary):
    decimal = 0
    length = len(binary)

    # fine decimal
    for i in range(length):
        decimal *= 2
        decimal += binary[i]

    #convert into ternary
    ternary = []
    for i in range(5):
        ternary.insert(0, decimal % 3)
        decimal = decimal //3

    return ternary

def ternary_to_binary(ternary):
    decimal = 0
    for i in range(5):
        decimal *= 3
        decimal += ternary[i]

    binary = []
    for i in range(8):
        binary.insert(0, decimal % 2)
        decimal = decimal // 2

    return binary

def encode(spec):
    #
    binary_code = []
    ternary_code = []

    for r in range(7):
        for c in range(r+1,7):
            binary_code.append(spec.matrix[r][c])

    for i in range(1,6):
        for idx in range(3):
            if su.ALLOWED_OPS[idx] == spec.ops[i]:
                ternary_code.append(idx)
                break

    binary_code.extend( ternary_to_binary(ternary_code) )
    return binary_code

def decode(binary_code):
    matrix = su.np.zeros((7,7))
    ops = [su.INPUT]

    idx = 0
    for r in range(7):
        for c in range(r+1,7):
            matrix[r][c] = binary_code[idx]
            idx+=1

    binary_code = binary_code[idx:]
    ternary_code = binary_to_ternary( binary_code )

    for idx in range(5):
        ops.append(su.ALLOWED_OPS[ ternary_code[idx] ])
    ops.append(su.OUTPUT)

    model_spec = su.api.ModelSpec( matrix, ops)
    return model_spec

def parent_selection(population, tournament_size):
    population_size = len(population)
    selected = []
    for _ in range(2):
        indexes= su.random.sample(range(population_size), tournament_size)
        pool = list(population[i] for i in indexes)
        pool.sort(key = crowded_comparison_operator)
        selected.append(pool[0])
    return selected

def crossover( parent1, parent2):
    index = su.random.sample(range(1,len(parent1)),1)
    offspring = parent1[:index]
    offspring.extend(parent2[index:])
    return offspring

def mutation(offspring, mutation_rate):
    index = su.random.sample(range(1,len(offspring)),1)
    # bitwise
    offspring[index] = ( 1 + offspring[index] ) % 2
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
        parent1, parent2  = parent_selection(population, tournament_size)

        parent1 = encode(parent1)
        parent2 = encode(parent2)

        # crossover
        offspring = parent1
        if su.random.random() < crossover_prob:
            offspring = crossover(parent1, parent2)

        # mutation
        if su.random.random() < mutation_rate:
            offspring = mutation(offspring)

        offspring_spec = decode(offspring)

        if su.nasbench.is_valid(offspring_spec):
            size += 1
            data = su.nasbench.query(offspring_spec)
            elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': offspring_spec}
            offspring_population.append(elem)

    return offspring_population

#def fast_non_dominated_sort(population):
#def crowding_distance_assignment():
#def crowded_comparison_operator( elem1, elem2 ):
    # elem is dictionary
    # return -1: elem1 < elem2  /   1: elem1 < elem2

def nsga2(answer_size=500,
          search_time=25000,
          population_size=500,
          crossover_prob=0.9,
          tournament_size=10,
          mutation_rate=0.03):

    su.nasbench.reset_budget_counters()
    # 2 objectives: validation, time
    population = [] # element is { rank, validation accuracy , time, spec } dictionary

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    for _ in range(population_size):
        spec = random_spec()
        data = su.nasbench.query(spec)
        elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': spec}
        population.append( elem )

    # evolution
    for _ in range(search_time):
        population += generate_offspring(population, crossover_prob, tournament_size, mutation_rate)
        fast_non_dominated_sort()
        number = 0





