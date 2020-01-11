import matplotlib.pyplot as plt
import nsgaNet
import overall_data
import random_pareto

''' 'NSGA-Net-crossover-0.1', 'NSGA-Net-crossover-0.5', 'NSGA-Net-crossover-1.0',
              'NSGA-Net-mutation-0.1', 'NSGA-Net-mutation-0.5', 'NSGA-Net-mutation-1.0' '''

'''
data = {'acc': , 'time': } list
'''


def pareto_front(data):
    answer_list = []
    for new_elem in data:
        append_flag = True
        idx = 0
        for elem in answer_list[:]:
            judge = nsgaNet.dominate_operator(elem, new_elem)
            if judge > 0:
                answer_list.remove(elem)
                idx -= 1
            elif judge < 0:
                append_flag = False
                break
        if append_flag:
            answer_list.append(new_elem)
    accuracy_list = [elem['acc'] for elem in answer_list]
    time_list = [elem['time'] for elem in answer_list]

    return accuracy_list, time_list


def main():
    acc = {}
    time = {}

    # plot1
    overall = overall_data.generate_all_data()
    acc['Answer'], time['Answer'] = pareto_front(overall)
    # len of 'Answer' = 44

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(time['Answer'], acc['Answer'], marker='s', color='violet', label='Answer')

    random = random_pareto.generate_random_data(240, axes[1])
    acc['Random'], time['Random'] = pareto_front(random)

    nsga = nsgaNet.generate_nsgaII_data(axes)
    acc['NSGA-Net'], time['NSGA-Net'] = pareto_front(nsga)

    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    plt.tight_layout()

    # plot2
    labels = ['Answer', 'Random', 'NSGA-Net']
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # yellow, light green, gray, violet, steel blue, green
    colors = ['#fff176', '#a5d6a7', '#9e9e9e', 'violet', 'red', 'green']
    markers = [ 'o', 'x', '*', '^', '>']
    ax.set_xlabel('time')
    ax.set_ylabel('accuracy')

    for i, label in enumerate(labels):
        ax.scatter(time[label], acc[label], marker=markers[i], color=colors[i], label=label)

    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

