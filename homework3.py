import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


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
print(ev) 

# Convert the principal components to a DataFrame
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9', 'PC-10'])

# Concatenate the principal components with the ground truth data
finalDf = pd.concat([principalDf, gt], axis = 1)

# Print the DataFrame
print(principalDf)

# Dimensionality reduction via PCA 
x1 = x.transpose()
X_pca = np.matmul(x1, principalComponents)
X_pca.shape 

# Model the dataframe 
x_pca_df = pd.DataFrame(data = X_pca, columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9', 'PC-10'])

# Add the labels 
X_pca_df = pd.concat([x_pca_df, gt], axis = 1)

# Print the dataframe
print(X_pca_df)

# Below is the data visualization (bar graph)
plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], list(ev * 100), label = 'Principal Components', color = 'b')
plt.legend()

plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')

pc = []
for i in range(10):
    pc.append('PC-' + str(i + 1))

plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], pc)
plt.title('Variance Ratio of INDIAN PINES Dataset')
plt.show()