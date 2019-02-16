"""
Plot the Loss or Accuracy of Both Train and Test
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_one(values, title):
    """plot one data

    Args:
        values: the data to plot
        title: the plot's title
    """
    plt.plot(values, label=title)
    plt.legend('best')
    plt.title(title.title())
    plt.xlabel('Epoch')
    plt.ylabel(title.split()[1].title())

def plot_two(data_dict,save_path):
    """plot the data_dict and save to png file

    Args:
        data_dict: A dict contains the data wanted to plot,
                   the keys should be the plot's title
                   the values should be a list to plot
    """
    assert len(data_dict) == 2, 'The function just plot 2 data'
    plt.figure(figsize=(15, 6))
    key1, key2 = data_dict.keys()
    plt.subplot(121)
    plot_one(data_dict[key1], key1)
    plt.subplot(122)
    plot_one(data_dict[key2], key2)
    plt.savefig(save_path + '/' + '{}.png'.format(key1.split()[1].title()))


def save_plot(data_dict,save_path):
    """save the plot of data

    Args:
        data_dict: A dict contains the data wanted to plot,
                   the keys should be the plot's title
                   the values should be a list to plot
    """
    plt.figure(figsize=(15, 6))
    for i, (key, value) in enumerate(data_dict.items()):
        plt.subplot(1, len(data_dict), i+1)
        plot_one(value, key)
    key1, *rest_key = data_dict.keys()
    plt.savefig(save_path + '/' + '{}.png'.format(key1.split()[1].title()))


def save_auc(auc_values, filename='AUG.png'):
    """plot tpr and fpr

    Args:
        auc_values: (auc, tpr, fpr)
                    tuple
    """
    auc, tpr, fpr = auc_values
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title('AUC')
    plt.text(0.75, 0.25, 'AUC = {:.4f}'.format(auc))
    plt.savefig(filename)
