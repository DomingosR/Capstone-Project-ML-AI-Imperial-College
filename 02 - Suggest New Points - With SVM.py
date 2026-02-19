# **********************************************************************
# Code to suggest new data points for experiments in Imperial College's
# capstone project.  This code is based on Bayesian Optimization
# using Upper Confidence Bounds as the objective function.  This Bayesian
# Optimization is preceded by the estimation of a SVM to determine which
# regions of the domain seem more promising for exploration, and focus
3 on those.
#
# The current version implements a dynamic value for kappa, the parameter
# that trades off exploration vs exploitation.  This parameter will change
# from one week to the next to emphasize greater exploitation over time.
#
# The core of this code is due to ChatGPT and Google Gemini.
#
# This version: 19 February 2026 
# Author: Domingos Romualdo
# **********************************************************************

import numpy as np
from sklearn.svm import SVC
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from scipy.optimize import minimize
from scipy.stats import norm
# from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import csv

# Import data and set up variables
variableCount = [2, 2, 3, 4, 4, 5, 6, 8]

# Reading input data
# This assumes input data is in .csv format
# Each row contains the experiment number, each of the 8 values for X
# (blank if not applicable), and the value for the corresponding Y
inputFile = "C:\\Users\\Domingos\\Documents\\Personal\\Machine Learning\\Capstone Project\Data\\Input.csv"

with open(inputFile, newline='') as file:
    reader = csv.reader(file)
    inputData = list(map(list, reader))

# Choosing kappa according to current week
weekNo = 4
kappa = 3 - weekNo * 0.125

# Variable to hold suggested values
suggestedValues  = []

# Loop to conduct experiments and get suggested values
for i in range(1, 9):
    # Filtering data
    filterData   = [row for row in inputData if row[0] == str(i)]
    dataPoints   = len(filterData)
    numVars      = variableCount[i - 1]
    bounds       = [(0.0, 1.0)] * variableCount[i - 1]

    # Creating variables in appropriate format for BO and SVM 
    X_init, y_init = [[] for _ in range(dataPoints)], []
    for rowNum in range(dataPoints):
        row      = filterData[rowNum]
        for j in range(1, numVars + 1):
            X_init[rowNum].append(float(row[j]))
        # Note that the experiment minimizes Y, but SVM needs a classification label where higher is better, so we negate Y here
        y_init.append(- float(row[-1]))

    # Define the Optimizer
    opt = Optimizer(
        dimensions = bounds,
        base_estimator = GaussianProcessRegressor(alpha=1e-6),
        acq_func = "LCB", # Upper Confidence Bound (Lower here for minimization)
        n_initial_points = 0
    )

    # Tell the optimizer about our existing data
    opt.tell(X_init, y_init)

    # Computing next sample for this experiment
    # For full sampling experiment, see Version 08 - From Google Gemini - With SVM.py

    # Train SVM on current "best" performers
    # We define 'best' as the top 25% of points found so far
    threshold = np.percentile(y_init, 25) 
    y_labels = (np.array(y_init) <= threshold).astype(int)
    
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_init, y_labels)

    # Hybrid Selection:
    # We ask BO for many candidates, but only pick the one SVM likes best
    candidates = opt.ask(n_points = 50) # Sample potential points
    
    # Get SVM's probability of being 'Good' (Class 1)
    probs = clf.predict_proba(candidates)[:, 1]
    
    # Get indices of top 20% candidates according to probs
    num_top_candidates = int(0.2 * len(probs))
    top_indices = np.argpartition(probs, -num_top_candidates)[-num_top_candidates:]
    probs = probs[top_indices]
    candidates = np.array(candidates)[top_indices]

    # Choosing best candidate on the basis of LCM (Lower Confidence Bound) criterion among the top candidates
    gp = opt.models[-1] 
    mu, std = gp.predict(candidates, return_std = True)
    
    # Calculate UCB (UCB = mu + kappa * std)
    # Since skopt internally minimizes, mu is often negated; adjust based on your setup.
    ucb_values = mu - (kappa * std) # Using - for LCB/Minimization context    
    best_idx = np.argmin(ucb_values)
    next_x = candidates[best_idx]

    # Append next_x to list of candidates
    suggestedValues.append(tuple(next_x))

print(suggestedValues)

