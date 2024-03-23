# Needed imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 components
X_pca = pca.fit_transform(X)

# Plot the data points in the space defined by the first two principal components
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)

plt.title('PCA of Iris dataset - First Two Principal Components')
plt.xlabel('PC-1')
plt.ylabel('PC-2')
plt.legend()
plt.grid(True)
plt.show()

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
pca = PCA(n_components = 10)
principalComponents = pca.fit_transform(x)

# Display contribution of each principal component
ev = pca.explained_variance_ratio_

# Dimensionality reduction via PCA 
x1 = x.transpose()
X_pca = np.matmul(x1, principalComponents)
X_pca.shape 

# Model the dataframe 
x_pca_df = pd.DataFrame(data = X_pca, columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9', 'PC-10'])

# Add the labels 
X_pca_df = pd.concat([x_pca_df, gt], axis = 1)

# Below is more visualization (scatter plot)
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('PC-1', fontsize = 15)
ax.set_ylabel('PC-2', fontsize = 15)

ax.set_title('PCA on INDIAN PINES Dataset', fontsize = 20)
class_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'r', 'g', 'b', 'y', 'm', 'c', 'k', 'b', 'r']
markerm = ['o', 'o', 'o', 'o', 'o', 'o', 'o', '+', '+', '+', '+', '+', '+', '+', '*', '*']

for target, color, m in zip(class_num, colors, markerm):
    indicesToKeep = X_pca_df['gth'] == target
    ax.scatter(X_pca_df.loc[indicesToKeep, 'PC-1'], X_pca_df.loc[indicesToKeep, 'PC-2'], c = color, s = 9, marker = m)

ax.legend(class_num)
ax.grid()  
plt.show()


# LDA on the IRIS dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

lda = LinearDiscriminantAnalysis(n_components = 2)
X_r2 = lda.fit(X, y).transform(X)

colors = ['navy', 'turquoise', 'darkorange']

plt.figure() 
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 0], alpha = .8, color = color, label = target_name)

plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
plt.title('LDA of IRIS dataset')

plt.show()