# EPFL Machine Learning - Project 2
# Option A
# Title: Spatially-Inferred Graphical Models for fMRI Data Mining
# Laboratory: Medical Image Processing Lab
# Supervised by Younes Farouj 
# Professor: Dimitri Van De Ville

In this document we present an overview of the project architecture and how to run it, as well as some useful information about the implementation. More specific information is provided in the project report.

## Project folders
* `data/`. Directory with the input data namely 'Activity_avr.mat', which contains the fMRI signals, and 'IC_AAL.mat', which contains the iCAPs. Additional files for visualization are provided 'CodeBook_90'.

* `report/`. Directory with the report in pdf `report.pdf` and in `report.tex` and the images included.
 
* `scripts/` Python files
    * `run.py`. Main file for estimating causality between brain regions.
    * `run_synthetic.py`. File for visualizing results with VAR synthetic data and testing different models with lags, nodes and lambdas.
    * `data_functions.py`. This script includes all the functions required to load, reshape and deal with data.
    * `graphic_functions.py`. This script includes functions required to visalize results.
    * `model_functions.py`. This script includes  the implementations for the different models considered.
    * `VAR.py`. This file generates the artificial data and creates the adjacency matrix, the C matrix, the noise generator and the final matrix required to test the model.

        MATLAB files (only for visualization purposes)
    * `visualize_brain.m`. In order to reproduce some of the brain images, we used a tool provided by the lab. When runing this file, the images of the report are reproduced.
    * `MatlabVisualization`: folder that contains additional files that are used in `visualize_brain.m`. 
    * `coefficients_90_lag_1.mat`: results used for the report
## Requirements for running the project

A valid installation of Python 3 is required.
The packages used are: numpy, matplotlib, seaborn, sklearn, scipy, os, networkx and bokeh.
For theirs installation, write in terminal:
* `pip3 install library`, where library is the name of the libraries cited above.

In addition, if the user desires to reproduce the brain images and the iCAPs, Matlab is required and the installation of the software FreeSurfer https://surfer.nmr.mgh.harvard.edu/ 
* In order to install the software, follow the instructions in https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall

## Running the project

Two main files are provided, `run.py` and `run_synthetic.py`, one for causality within the brain and the other for causality using synthetic data. In both, the results that are shown in the report are reproduced. 
In order to visualize some of the images in the report a matlab tool provided by the lab was used. Run `visualize_brain.m` file in order to get the brain images.

