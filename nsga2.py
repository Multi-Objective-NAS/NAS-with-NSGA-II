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

def encode(spec):
    binary_code = []

    return binary_code

def decode(binary_code):

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

def mutation(offspring, mutation_rate):

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

def fast_non_dominated_sort(population):
def crowding_distance_assignment():
def crowded_comparison_operator( elem1, elem2 ):
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
        while number <= population_size:





