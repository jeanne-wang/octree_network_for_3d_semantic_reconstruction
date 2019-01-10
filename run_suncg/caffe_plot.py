# In the name of GOD the most compassionate the most merciful
# Originally developed by Yasse Souri
# Just added the search for current directory so that users dont have to use command prompts anymore!
# and also shows the top 4 accuracies achieved so far, and displaying the highest in the plot title 
# Coded By: Seyyed Hossein Hasan Pour (Coderx7@gmail.com)
# -------How to Use ---------------
# 1.Just place your caffe's traning/test log file (with .log extension) next to this script
# and then run the script.If you have multiple logs placed next to the script, it will plot all of them
# you may also copy this script to your working directory, where you generate/keep your train/test logs
# and easily execute the script and see the curve plotted. 
# this script is standalone.
# 2. you can use command line arguments as well, just feed the script with different log files separated by space
# and you are good to go.
#----------------------------------
import numpy as np
import re
import click
import glob, os
import matplotlib
matplotlib.use('GTKAgg')  #for remote plot
from matplotlib import pylab as plt

import operator
import ntpath
@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    
    for i, log_file in enumerate(files):
        loss_iterations, losses, accuracy_iterations, freespace_accuracies, overall_accuracies, semantic_accuracies, fileName = parse_log(log_file)
        disp_results(loss_iterations, losses, accuracy_iterations, freespace_accuracies, 
                    overall_accuracies, semantic_accuracies, fileName)


def parse_log(log_file):
    with open(log_file, 'r') as log_file2:
        log = log_file2.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+).*, loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    fileName= os.path.basename(log_file)
    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    freespace_accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* freespace_accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    overall_accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.*\n.* overall_accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    semantic_accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.*\n.*\n.*\n.*\n.*\n.*\n.*\n.* semantic_accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    freespace_accuracies = []
    overall_accuracies = []
    semantic_accuracies = []
    accuracy_iterations = []
    for r in re.findall(freespace_accuracy_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        accuracy_iterations.append(iteration)
        freespace_accuracies.append(accuracy)

    accuracy_iterations = np.array(accuracy_iterations)
    freespace_accuracies = np.array(freespace_accuracies)

    for r in re.findall(overall_accuracy_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        overall_accuracies.append(accuracy)

    overall_accuracies = np.array(overall_accuracies)

    for r in re.findall(semantic_accuracy_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        semantic_accuracies.append(accuracy)

    semantic_accuracies = np.array(semantic_accuracies)

    assert(len(semantic_accuracies)==len(freespace_accuracies))
    assert(len(overall_accuracies)==len(freespace_accuracies))

    
    return loss_iterations, losses, accuracy_iterations, freespace_accuracies, overall_accuracies, semantic_accuracies, fileName


def disp_results(loss_iterations, losses, accuracy_iterations, freespace_accuracies, 
                overall_accuracies, semantic_accuracies, fileName):

    plt.style.use('ggplot')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax1.plot(loss_iterations, losses, color='r')

    ax2.set_xlabel('iteration')
    ax2.set_ylabel('freespace_accuracy %')
    ax2.plot(accuracy_iterations, freespace_accuracies, 'g')


    ax3.set_xlabel('iteration')
    ax3.set_ylabel('overall_accuracy %')
    ax3.plot(accuracy_iterations, overall_accuracies,'m')

    ax4.set_xlabel('iteration')
    ax4.set_ylabel('semantic_accuracy %')
    ax4.plot(accuracy_iterations, semantic_accuracies, 'b', label=str(fileName))
   
    plt.legend(loc='lower right') 
    plt.show()


if __name__ == '__main__':
    main()