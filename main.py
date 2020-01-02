import matplotlib.pyplot as plt
import nsgaNet
import pareto_front
import random_pareto


def main():
    acc = {}
    time = {}

    acc['NSGA-Net'], time['NSGA-Net'] = nsgaNet.nsgaII()
    acc['Total-data'], time['Total-data'] = pareto_front.pareto_front(False)
    acc['Answer'], time['Answer'] = pareto_front.pareto_front(True)
    acc['Random'], time['Random'] = random_pareto.random_pareto_front()

    labels = ['NSGA-Net', 'Total-data', 'Answer', 'Random']
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    colors = ['green', 'yellow', 'steelblue', 'violet']
    markers = ['o', 'x', '*', '^']

    for i, label in enumerate(labels):
        ax.scatter(time[label], acc[label], marker=markers[i], color=colors[i], label=label)

    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()
