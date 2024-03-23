from scipy.io import loadmat
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Mode 1: PCA on the Indian Pines dataset
# Mode 2: Covariance matrix visualization example
# Mode 3: LDA on the IRIS dataset
# Mode 4: Iris classification (Not working) 
mode = 2

if(mode == 1):
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

if(mode == 2):
    # Covariance matrix visualization example
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (12, 8)

    # Generate normal distribution data
    x1 = np.random.normal(0, 1, 1000)
    x2 = np.random.normal(0, 1, 1000)
    x3 = np.random.normal(0, 1, 1000)

    X = np.vstack((x1, x2, x3)).T

    plt.scatter(X[:, 0], X[:, 1])
    plt.title('Normal distribution data N(0,1) - X')
    plt.axis('equal')

    plt.show()

    # Calculate the covariance matrix and visualize it
    X_df = pd.DataFrame(X, columns = ['1', '2', '3'])
    cov_matrix = np.cov(X_df.T, bias = True)
    print(cov_matrix)

    sn.heatmap(cov_matrix, annot = True)
    plt.show() 

    # Find the principle components of the covariance matrix
    pca = PCA(n_components = 2)
    PCs = pca.fit_transform(X.T)

    print('The PCs for data X generated = \n', PCs)

    # Display contribution of each pc 
    ev = pca.explained_variance_ratio_

    print('The explained variance ratio = \n', (ev * 100))

    # Plot variance/pc
    plt.bar([1, 2], list(ev * 100), label = 'Principal Components', color = 'b')
    plt.legend()

    plt.xlabel('Principal Components')
    pc = []

    for i in range(2):
        pc.append('PC-' + str(i + 1))

    plt.xticks([1, 2], pc, fontsize = 8, rotation = 30)
    plt.ylabel('Variance Ratio')

    plt.title('Variance Ratio of Normal Distribution Data X')
    plt.show() 

    # PCA transformed data 
    Y = np.matmul(X, PCs)

    # PCA visualization
    # Plot eig values and their vectors
    C = np.cov(Y.T)

    eig_vec, eig_val = np.linalg.eig(C)
    print('The eigen values = \n', eig_val)
    print('The eigen vectors = \n', eig_vec)

    plt.scatter(Y[:, 0], Y[:, 1])

    for e, v in zip(eig_vec, eig_val.T):
        plt.plot([0, 5 * np.sqrt(e) * v[0]], [0, 5 * np.sqrt(e) * v[1]], 'k-', lw = 2)
    
    plt.title('PCA Transformed Data Y = T(X)')
    plt.axis('equal')

    plt.show() 

if(mode == 3):

    # LDA on the IRIS dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    pca = PCA(n_components = 2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components = 2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color = color, alpha = .8, lw = lw, label = target_name)

    plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
    plt.title('PCA of IRIS dataset')

    plt.figure() 
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 0], alpha = .8, color = color, label = target_name)

    plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
    plt.title('LDA of IRIS dataset')

    plt.show()

if(mode == 4):

    # Iris classification 
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Split the data into training and test sets
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size = 0.20, random_state = 1, shuffle = True)

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma = 'auto')))

    # Evaluate each model in turn
    results = []
    names = []

    for name, model in models:
        kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
        cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean() * 100, cv_results.std()))

    # Compare Algorithms
    plt.boxplot(results, labels = names)
    plt.title('Algorithm Comparison')
    plt.show()

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
    axes[0].set_title("Model Evaluation: Training/Recall Accuracy")
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





