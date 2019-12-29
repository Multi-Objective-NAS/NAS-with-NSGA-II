import matplotlib.pyplot as plt
import nsgaNet
import _pareto_front


def main():
    acc = {}
    time = {}

    acc['NSGA-Net'], time['NSGA-Net'] = nsgaNet.nsgaII()
    #acc['Answer'], time['Answer'] = _pareto_front.pareto_front()

    labels = ['NSGA-Net']
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    colors = ['salmon', 'steelblue']
    markers = ['o', 'x']

    for i, label in enumerate(labels):
        ax.scatter(acc[label], time[label], marker=markers[i], color=colors[i], label=label)

    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()
