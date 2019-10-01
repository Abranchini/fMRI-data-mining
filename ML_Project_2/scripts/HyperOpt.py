# This file is the main script to run the optimization of parameters for the elastic net
# but can be adapted to other functions

# general imports
import numpy as np ; import csv
import GPyOpt ; import GPy
import sklearn ; import importlib ; import time
import matplotlib.pyplot as plt ; import seaborn as sns

# This must be changed according to ones directory
import os ; import sys
dir_path = os.path.dirname(os.path.realpath("__file__"))
sys.path.append(dir_path + "/ML_Project_2/scripts/")

# function imports
import data_functions, model_functions, graphic_functions, VAR
# Reupdate functions
importlib.reload(VAR)
importlib.reload(data_functions)
importlib.reload(model_functions)
importlib.reload(graphic_functions)

# This is the main file for getting the causality in the brain.
from VAR import *
from data_functions import *
from model_functions import *
from graphic_functions import *

# initialize values
P = 1
N = 6
T = 100
L = 4
lambda_1 = 0.3 ; lambda_2 = 0.9
y_output = [] ; acquisition_type = "LCB"

iter_count = 400
current_iter = 0
X_step = np.array([[0.1, 0.9]])

# make tests using ridge
acc, std = hypeOpt_elastic_synth(P,N,T,L, 20, lambda_1, lambda_2)

yR = acc
y_output.append(yR)
Y_step = np.reshape(y_output,(-1,1))
domain =[{'name': 'lambda_1', 'type': 'continuous', 'domain': (0.1,3)},\
    {'name': 'lambda_2', 'type': 'continuous', 'domain': (0.1,3)}]


file_directory = "/media/abranches/Main/University/EPFL/Project_MPI/fMRI-data-mining/ML_Project_2/tests/"
# Windows save directory
#file_directory = "D:/University/EPFL/Project_MPI/fMRI-data-mining/ML_Project_2/tests/"

# open file to write tests
date = time.strftime("%Y-%m-%d_%H_%M_%S",time.gmtime())
with open(file_directory + date + "_" + acquisition_type + ".csv", "w") as OV_file:
    writer = csv.DictWriter(OV_file, fieldnames = ["lambda_1", "lambda_2","y"])

    # writer = csv.DictWriter(OV_file, fieldnames = ["lengthscale","variance","y"])

    writer.writeheader()


while current_iter < iter_count:
    start = time.time()
    bo_step = GPyOpt.methods.BayesianOptimization(f = None, domain = domain, X = X_step, Y = Y_step, 
                                            acquisition_type=acquisition_type,
                                            maximize = True,exploration_weight=1000,exact_feval=True)
    
    x_next = bo_step.suggest_next_locations()
    end = time.time()
    print("Time to suggest new location {}".format(end-start))

    lambda_1 = x_next[0][0]
    lambda_2 = x_next[0][1]

    start = time.time()
    try:
        acc , std= hypeOpt_elastic_synth(P,N,T,L, 20, lambda_1, lambda_2)
        end = time.time()
        print("Time to run {}".format(end-start))

        yR = acc
        yR_write = yR
    except Exception as e:
        yR = np.inf
        yR_write = e

    y_next = yR
    print("\n Iteration : {}/{}. lamba_1 :{}. lambda_2 : {}. Accuracy : {}".format(\
        current_iter, iter_count , lambda_1, lambda_2, acc))

    X_step = np.vstack((X_step, x_next))
    Y_step = np.vstack((Y_step, y_next))

    with open(file_directory + date + "_" + acquisition_type + ".csv", "a") as OV_file:
        writer = csv.writer(OV_file)
        writer.writerows( [[lambda_1, lambda_2, yR_write , std]]) 

    current_iter +=1


