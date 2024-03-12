from scipy.io import loadmat
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets 

# Load the data
df = loadmat('indianR.mat')

x = np.array(df['X'])

gth = np.array(df['gth'])

num_rows = np.array(df['num_rows'])

num_cols = np.array(df['num_cols'])

num_bands = np.array(df['num_bands'])

bands, samples = x.shape

# Load the ground truth data 
gth_mat = loadmat('indian_gth.mat')
gth_mat = {i : j for i, j in gth_mat.items() if i[0] != '_'}
gt = pd.DataFrame({i : pd.Series(j[0]) for i, j in gth_mat.items()})
