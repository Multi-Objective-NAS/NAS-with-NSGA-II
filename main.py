import matplotlib.pyplot as plt
import nsgaNet
import pareto_front
import random_pareto

''' 'NSGA-Net-crossover-0.1', 'NSGA-Net-crossover-0.5', 'NSGA-Net-crossover-1.0',
              'NSGA-Net-mutation-0.1', 'NSGA-Net-mutation-0.5', 'NSGA-Net-mutation-1.0' '''


def main():
    acc = {}
    time = {}

    acc['Random'], time['Random'] = random_pareto.random_pareto_front(640)
    acc['NSGA-Net'], time['NSGA-Net'], acc['Search Space'], time['Search Space'] = nsgaNet.nsgaII()
    acc['Total-data'], time['Total-data'], acc['Answer'], time['Answer'] = pareto_front.pareto_front()

    labels = ['Total-data','Search Space', 'Answer', 'Random', 'NSGA-Net']
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # yellow, gray, violet, steel blue, green
    colors = ['#fff176', '#9e9e9e', 'violet', 'red', 'green']
    markers = ['o', 'x', '*', '^', '>']

    for i, label in enumerate(labels):
        ax.scatter(time[label], acc[label], marker=markers[i], color=colors[i], label=label)

    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()
