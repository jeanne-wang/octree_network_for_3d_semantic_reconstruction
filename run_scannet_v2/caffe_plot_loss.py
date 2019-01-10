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
        loss_iterations, prop_losses_l0, prop_losses_l1, seg_losses_l0, seg_losses_l1, seg_losses_l2, fileName = parse_log(log_file)
        disp_results(loss_iterations, prop_losses_l0, prop_losses_l1, seg_losses_l0, seg_losses_l1, seg_losses_l2, fileName)


def parse_log(log_file):
    with open(log_file, 'r') as log_file2:
        log = log_file2.read()

    fileName= os.path.basename(log_file)
    ## prop_loss_l0
    prop_loss_l0_pattern = r"Iteration (?P<iter_num>\d+).*\n.*Train net output #0: prop_loss_l0 = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?) (.*)"
    prop_losses_l0 = []
    loss_iterations = []
    
    for r in re.findall(prop_loss_l0_pattern, log):
        loss_iterations.append(int(r[0]))
        prop_losses_l0.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    prop_losses_l0 = np.array(prop_losses_l0)

    ## prop_loss_l1
    prop_loss_l1_pattern = r"Iteration (?P<iter_num>\d+).*\n.*\n.*Train net output #1: prop_loss_l1 = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?) (.*)"
    prop_losses_l1 = []

    for r in re.findall(prop_loss_l1_pattern, log):
        prop_losses_l1.append(float(r[1]))

    prop_losses_l1 = np.array(prop_losses_l1)

    ## seg_loss_l0
    seg_loss_l0_pattern = r"Iteration (?P<iter_num>\d+).*\n.*\n.*\n.*Train net output #2: seg_loss_l0 = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?) (.*)"
    seg_losses_l0 = []

    for r in re.findall(seg_loss_l0_pattern, log):
        seg_losses_l0.append(float(r[1]))

    seg_losses_l0 = np.array(seg_losses_l0)

    ## seg_loss_l1
    seg_loss_l1_pattern = r"Iteration (?P<iter_num>\d+).*\n.*\n.*\n.*\n.*Train net output #3: seg_loss_l1 = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?) (.*)"
    seg_losses_l1 = []

    for r in re.findall(seg_loss_l1_pattern, log):
        seg_losses_l1.append(float(r[1]))

    seg_losses_l1 = np.array(seg_losses_l1)

    ## seg_loss_l2
    seg_loss_l2_pattern = r"Iteration (?P<iter_num>\d+).*\n.*\n.*\n.*\n.*\n.*Train net output #4: seg_loss_l2 = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?) (.*)"
    seg_losses_l2 = []

    for r in re.findall(seg_loss_l2_pattern, log):
        seg_losses_l2.append(float(r[1]))

    seg_losses_l2 = np.array(seg_losses_l2)

    iterations = len(loss_iterations)
    print(iterations)
    #assert(len(prop_losses_l0)==iterations)
    #assert(len(prop_losses_l1)==iterations)
    #assert(len(seg_losses_l0)==iterations)
    #assert(len(seg_losses_l1)==iterations)
    #assert(len(seg_losses_l2)==iterations)


    
    return loss_iterations, prop_losses_l0, prop_losses_l1, seg_losses_l0, seg_losses_l1, seg_losses_l2, fileName


def disp_results(loss_iterations, prop_losses_l0, prop_losses_l1, seg_losses_l0, seg_losses_l1, seg_losses_l2, fileName):

    plt.style.use('ggplot')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('prop_loss_l0')
    ax1.plot(loss_iterations, prop_losses_l0, color='r')

    ax2.set_xlabel('iteration')
    ax2.set_ylabel('prop_loss_l1')
    ax2.plot(loss_iterations, prop_losses_l1, 'g')


    ax4.set_xlabel('iteration')
    ax4.set_ylabel('seg_loss_l0')
    ax4.plot(loss_iterations, seg_losses_l0,'m')

    ax5.set_xlabel('iteration')
    ax5.set_ylabel('seg_loss_l1')
    ax5.plot(loss_iterations, seg_losses_l1, 'b')

    ax6.set_xlabel('iteration')
    ax6.set_ylabel('seg_loss_l2')
    ax6.plot(loss_iterations, seg_losses_l2, 'c', label=str(fileName))

   
    plt.legend(loc='upper left') 
    plt.show()


if __name__ == '__main__':
    main()
