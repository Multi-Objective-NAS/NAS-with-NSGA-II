import constants
import nsgaNet
import random


def generate_random_data(size, ax):
    random_list = []
    constants.nasbench.reset_budget_counters()
    # element = {'acc': , 'time': }
    times, hvs = nsgaNet.random_generation(random_list, size)
    ax.plot(times, hvs, label='Random', color='red')

    return random_list
