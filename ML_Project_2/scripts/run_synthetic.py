# general imports
import sys
import sklearn
import importlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#sys.path.append('D:/University/EPFL/Project_MPI/ML_Project_2/scripts')
sys.path.append('/media/abranches/Main/University/EPFL/Project_MPI/ML_Project_2/scripts')
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

# define lambdas to use
lambda_list = np.arange(0,1,0.05)

# initialize values
P = 1
N = 4
T = 100
L = 3

# make tests using ridge
RegStd, RegMetrics, RegAcc = synthetic_test(P,N,T,L, number_of_tests = 20, 
                                lambda_list = lambda_list ,model = ridgeRegression)

# make tests using lasso
# LaStd ,LaMetrics, LaAcc = synthetic_test(P,N,T,L, number_of_tests = 5, 
#                                 lambda_list = lambda_list ,model = lassoRegression)

# get x-axis labels
legend = []
for l in lambda_list:
    str_float = str(round(l,2))
    legend.append(str_float)

## visualization
# plot figure size
plt.figure(figsize=(20,10))
# plot bars
sns.barplot(legend,RegAcc,color='g',alpha = 0.5)#,label = "Ridge Regression")
# sns.barplot(legend,LaAcc,color='b',alpha = 0.5)#, label = "Lasso Regression")
# plot erros
# plt.errorbar(x=legend,y=LaAcc , yerr=LaStd, xerr=None,color='b')
plt.errorbar(x=legend,y=RegAcc , yerr=RegStd, xerr=None,color='g')

# define design of image
plt.xlabel('Lambdas value',size = 20)
plt.xticks(legend,rotation=90,size=15)

plt.ylabel('Accuracy',size = 20)
plt.yticks(size=15)
plt.ylim(0,1)

plt.legend(loc=2, prop={'size': 20})
plt.show()

