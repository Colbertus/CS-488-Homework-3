# Needed imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
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

# List features 
n = []
ind = [] 
for i in range(bands):
    n.append(i + 1)

for i in range(bands):
    ind.append('band' + str(n[i]))

features = ind

# Normalize the features (preprocessing)
# 'MinMaxScaler' is a form of normalization that scales the data to a range between 0 and 1
scaler_model = MinMaxScaler()
scaler_model.fit(x.astype(float))
x = scaler_model.transform(x) 

# Apply PCA to the normalized features
pca = PCA()
principalComponents = pca.fit_transform(x)

# Display contribution of each principal component
ev = pca.explained_variance_ratio_
print(ev) 

# Dimensionality reduction via PCA 
x1 = x.transpose()
X_pca = np.matmul(x1, principalComponents)
X_pca.shape 

# Below is the data visualization (bar graph)
plt.bar(np.arange(1, len(ev) + 1), list(ev * 100), label = 'Principal Components', color = 'b')
plt.legend()

# Label the x-axis and y-axis
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')

# Label the x-axis with the principal components
pc = []
for i in range(len(ev)):
    pc.append('PC-' + str(i + 1))

plt.xticks(np.arange(1, len(ev) + 1), pc)

# Title and display the graph 
plt.title('Variance Ratio of INDIAN PINES Dataset') 
plt.plot
plt.show()

# Load the IRIS dataset
iris = datasets.load_iris()
X = iris.data

# Apply PCA to the IRIS dataset
pca = PCA()
pca.fit(X)

# Display the contribution of each principal component
ev = pca.explained_variance_ratio_
plt.bar(np.arange(1, len(ev) + 1), list(ev * 100), label = 'Principal Components', color = 'b')  
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')

# Label the x-axis with the principal components
pc = []
for i in range(len(ev)):
    pc.append('PC-' + str(i + 1))

# Label the x-axis 
plt.xticks(np.arange(1, len(ev) + 1), pc)

# Title and display the graph
plt.showgrid = True
plt.title('Variance Ratio of IRIS Dataset')
plt.plot
plt.show() 

''' Skip this for now
# Reload the IRIS dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Apply PCA to the IRIS dataset
pca = PCA(n_components = 2)
PCs = pca.transform(X)
'''

