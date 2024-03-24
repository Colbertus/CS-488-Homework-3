# Needed imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix

#######################################################################

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

#######################################################################

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

#######################################################################

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

#######################################################################

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

#######################################################################


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

#######################################################################

# reduce the data to the desired number of components using LDA
def lda_reduction(data, gth, desiredComponents):
    # initialize the LDA model
    lda = LinearDiscriminantAnalysis(n_components=desiredComponents)
    
    # fit the model and transform the dataset
    linearDiscriminants = lda.fit_transform(data,gth)
    
    # create a DataFrame for the reduced components
    columns = ['LD-' + str(i) for i in range(1, desiredComponents + 1)]
    principalDf = pd.DataFrame(data=linearDiscriminants, columns=columns)
    
    # concatenate the linear discriminants with the ground truth labels
    finalDf = pd.concat([principalDf, gth], axis=1)
    
    return linearDiscriminants, finalDf

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

desiredComponents = 10

# normalize the data between 0 and 1
scaler_model = MinMaxScaler()
scaler_model.fit(x.astype(float))
x = scaler_model.transform(x) 

# apply LDA to the dataset using the function
linearDiscriminants, finalDf = lda_reduction(x.T, gth, desiredComponents)

# plot the first two directions in LDA with respect to color-coded class separability
plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

for target, marker in zip(np.unique(gth), markers):
    indicesToKeep = finalDf['gth'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'LD-1'], finalDf.loc[indicesToKeep, 'LD-2'], s=10, marker=marker, label=target)
    
ax.set_xlabel('LD-1', fontsize=15)
ax.set_ylabel('LD-2', fontsize=15)
ax.set_title('2 Component LDA for Indian Pines', fontsize=20)
ax.legend(np.unique(gth))
ax.grid()
plt.show()

#######################################################################


# Iris classification 
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split the data into training and test sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size = 0.20, random_state = 1, shuffle = True)

def plot_learning_curve(classifier, X, y, steps = 10, train_sizes = np.linspace(0.1, 1.0, 10), label = "", color = "r", axes = None):
    estimator = Pipeline([("scaler", MinMaxScaler()), ("classifier", classifier)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
    train_scores = [] 
    test_scores = []
    train_sizes = []
    
    for i in range(0, X_train.shape[0], X_train.shape[0] // steps):

        if (i == 0):
            continue 

        X_train_i = X_train[0 : i, :]
        y_train_i = y_train[0 : i]
        estimator.fit(X_train_i, y_train_i)
        train_scores.append(estimator.score(X_train_i, y_train_i) * 100)
        test_scores.append(estimator.score(X_test, y_test) * 100)
        train_sizes.append(i + 1)
    
    if (X_train.shape[0] % steps != 0):
        estimator.fit(X_train, y_train)
        train_scores.append(estimator.score(X_train, y_train) * 100)
        test_scores.append(estimator.score(X_test, y_test) * 100)
        train_sizes.append(X_train.shape[0])
    
    if axes is None:
        _, axes = plt.subplot(2)

    train_s = np.linspace(10, 100, num = 5)

    axes[0].plot(train_sizes, train_scores, 'o-', color = color, label = label)
    axes[1].plot(train_sizes, test_scores, 'o-', color = color, label = label)

    print("Training Accuracy of", label, ": ", train_scores[-1], "%")
    print("Testing Accuracy of", label, ": ", test_scores[-1], "%")
    print("")

    return plt 

# Create a model
_, axes = plt.subplots(1, 2, figsize = (12, 5))
num_steps = 10
classifier_labels = {
                    "Logistic Regression": (LogisticRegression(max_iter = 1000, random_state = 1), "red"),
                    "Random Forest": (RandomForestClassifier(random_state = 1), "green"),
                    "SVM = Linear": (SVC(kernel = 'linear', random_state = 1), "blue"),
                    "SVM = RBF": (SVC(kernel = 'rbf', random_state = 1), "yellow"),
                    "SVM = Poly": (SVC(kernel = 'poly', random_state = 1), "orange"),
                    "kNN": (KNeighborsClassifier(n_neighbors = 5), "purple"),
                    "Gaussian Naive Bayes": (GaussianNB(), "lime")
                    }

for label in classifier_labels: 
    classifier = classifier_labels[label][0]
    color = classifier_labels[label][1]
    plot_learning_curve(classifier, X, y, steps = num_steps, color = color, label = label, axes = axes)

axes[0].set_xlabel("% of Training examples")
axes[0].set_ylabel("Overall Classification Accuracy")
axes[0].set_title("Model Evaluation: IRIS dataset Training/Recall Accuracy")
axes[0].legend() 

axes[1].set_xlabel("% of Training examples")
axes[1].set_ylabel("Training/Recall Accuracy")
axes[1].set_title("Model Evaluation: Cross-Validation Accuracy")
axes[1].legend()

plt.show() 

def plot_per_class_accuracy(classifier, X, y, label, feature_selection = None):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state = 101)
    pipeline = Pipeline([("scaler", MinMaxScaler()), ("classifier", classifier)])
    pipeline.fit(X_train, Y_train)
    disp = confusion_matrix(pipeline, X_test, y_test, cmap = plt.cm.Blues)

    plt.title(label)
    plt.show()

    true_positive = disp.confusion_matrix[1][1]
    false_negative = disp.confusion_matrix[1][0]

    print(label + " - Sensitivity: ", true_positive / (true_positive + false_negative))
    print()

for label in classifier_labels:
    classifier = classifier_labels[label][0]
    plot_per_class_accuracy(classifier, X, y, label)




