import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# Iris Dataset Retrieval
def read_data(path):
    content = pd.read_csv(path)
    content.to_csv()
    lbl = LabelEncoder()
    lbl.fit(content["Class"].values)
    content["Class"] = lbl.transform(list(content["Class"].values))
    return content


# Plotting Data After Training and Testing
def plot_data(dataset, feature1, feature2, class1, class2, W, bias):
    if class1 == 'C1' and class2 == 'C2' or class2 == 'C1' and class1 == 'C2':
        data1 = dataset.iloc[:50, :]
        data2 = dataset.iloc[50:100, :]
    elif class1 == 'C1' and class2 == 'C3' or class2 == 'C1' and class1 == 'C3':
        data1 = dataset.iloc[:50, :]
        data2 = dataset.iloc[100:150, :]
    else:
        data1 = dataset.iloc[50:100, :]
        data2 = dataset.iloc[100:150, :]
    dataset = pd.concat([data1, data2])
    groups = dataset.groupby("Class")
    for name, group in groups:
        if feature1 == "X1" and feature2 == "X2":
            plt.plot(group.X1, group.X2, marker='o', linestyle='', markersize=6, label=name)
        elif feature1 == "X1" and feature2 == "X3":
            plt.plot(group.X1, group.X3, marker='o', linestyle='', markersize=6, label=name)
        elif feature1 == "X1" and feature2 == "X4":
            plt.plot(group.X1, group.X4, marker='o', linestyle='', markersize=6, label=name)
        elif feature1 == "X2" and feature2 == "X3":
            plt.plot(group.X2, group.X3, marker='o', linestyle='', markersize=6, label=name)
        elif feature1 == "X2" and feature2 == "X4":
            plt.plot(group.X2, group.X4, marker='o', linestyle='', markersize=6, label=name)
        elif feature1 == "X3" and feature2 == "X4":
            plt.plot(group.X3, group.X4, marker='o', linestyle='', markersize=6, label=name)
    x = np.linspace(0, 10, 100)
    if len(W) == 3:
        y = -(W[1] * x + W[0]) / W[2]
    elif len(W) == 2:
        y = -(W[0] * x) / W[1]
    plt.plot(x, y, '-r', label='y=-(w1*x1+b)/w2')
    plt.legend()
    plt.show()


# Perceptron Algorithm Model
def Perc_ALG(dataset, feature1, feature2, class1, class2, l_rate, epochs, bias):
    epochs = int(epochs)
    l_rate = float(l_rate)

    # Dataset Splitting Training/Testing
    if (class1 == 'C1' and class2 == 'C2') or (class2 == 'C1' and class1 == 'C2'):
        data1 = dataset.iloc[0:50, :]
        data2 = dataset.iloc[50:100, :]
    elif (class1 == 'C1' and class2 == 'C3') or (class2 == 'C1' and class1 == 'C3'):
        data1 = dataset.iloc[0:50, :]
        data2 = dataset.iloc[100:, :]
    else:
        data1 = dataset.iloc[50:100, :]
        data2 = dataset.iloc[100:, :]
    data1 = data1.sample(frac=1).reset_index(drop=True)
    data2 = data2.sample(frac=1).reset_index(drop=True)
    features = [feature1, feature2]
    X1 = data1[features]
    X2 = data2[features]
    Y1 = data1['Class']
    Y2 = data2['Class']
    xtrain1 = X1.iloc[:30]
    xtest1 = X1.iloc[30:]
    xtrain2 = X2.iloc[:30]
    xtest2 = X2.iloc[30:]
    ytrain1 = Y1.iloc[:30]
    ytest1 = Y1.iloc[30:]
    ytrain2 = Y2.iloc[:30]
    ytest2 = Y2.iloc[30:]
    xtr = pd.concat([xtrain1, xtrain2])
    xtst = pd.concat([xtest1, xtest2])
    ytr = pd.concat([ytrain1, ytrain2])
    ytst = pd.concat([ytest1, ytest2])
    labels1 = ytr.unique()
    labels2 = ytst.unique()
    ytr = ytr.replace({labels1[0]: 1, labels1[1]: -1})
    ytst = ytst.replace({labels2[0]: 1, labels2[1]: -1})
    xtr = np.array(xtr)
    xtst = np.array(xtst)
    if bias == 'Yes':
        xtr = np.insert(xtr, 0, np.ones([1, 1]), axis=1)
        xtst = np.insert(xtst, 0, np.ones([1, 1]), axis=1)
    ypred = np.array(int)

    # Training Start
    W = intialize(bias)
    xtr = np.array(xtr)
    ytr = np.array(ytr)
    W = nn_model_perceptron(xtr, ytr, W, epochs, l_rate)

    # Testing on Weights Retrieved from NN_Model
    xtst = np.array(xtst)
    for row in range(xtst.shape[0]):
        Z, A = forward(np.expand_dims(xtst[row], axis=1), W)
        ypred = np.append(ypred, A)
    error = 0
    inCorrectC1 = 0
    inCorrectC2 = 0
    ytst = np.array(ytst)
    for i in range(ytst.shape[0]):
        if ytst[i] != ypred[i]:
            error += 1
            if ytst[i] == 1:
                inCorrectC1 += 1
            else:
                inCorrectC2 += 1
    correctC1 = 20 - inCorrectC1
    correctC2 = 20 - inCorrectC2
    confusionMatrix = [[correctC1, inCorrectC1], [correctC2, inCorrectC2]]
    accuracy = ((correctC1 + correctC2) / 40) * 100

    # Confusion Matrix and Accuracy
    print(confusionMatrix)
    print("Error = ", error)
    print("Overall Accuracy = " + str(accuracy) + "%")
    return W, bias


# Signum Activation Function
def signum(Z):
    if Z[0][0] == 0:
        return 0
    elif Z[0][0] < 0:
        return -1
    else:
        return 1


# Initialization of Weights
def intialize(bias):
    if bias == 'No':
        W = np.random.rand(2, 1)
    else:
        W = np.random.rand(3, 1)
    return W


# Evaluation of Sample(i)
def forward(X, W):
    Z = np.dot(W.T, X)
    A = signum(Z)
    return Z, A


# Neural Network Model Algorithm
def nn_model_perceptron(X, Y, W, num_of_iterations, learning_rate):
    for i in range(num_of_iterations):
        for row in range(X.shape[0]):
            Z, A = forward(np.expand_dims(X[row], axis=1), W)
            if A != Y[row]:
                error = Y[row] - A
                W = W + learning_rate * error * np.expand_dims(X[row], axis=1)
    return W


def adaline_algo(dataset, feature1, feature2, class1, class2, l_rate, epochs, bias, threshold):
    epochs = int(epochs)
    l_rate = float(l_rate)
    threshold = float(threshold)
    # Dataset Splitting Training/Testing
    if (class1 == 'C1' and class2 == 'C2') or (class2 == 'C1' and class1 == 'C2'):
        data1 = dataset.iloc[0:50, :]
        data2 = dataset.iloc[50:100, :]
    elif (class1 == 'C1' and class2 == 'C3') or (class2 == 'C1' and class1 == 'C3'):
        data1 = dataset.iloc[0:50, :]
        data2 = dataset.iloc[100:, :]
    else:
        data1 = dataset.iloc[50:100, :]
        data2 = dataset.iloc[100:, :]
    data1 = data1.sample(frac=1).reset_index(drop=True)
    data2 = data2.sample(frac=1).reset_index(drop=True)
    features = [feature1, feature2]
    X1 = data1[features]
    X2 = data2[features]
    Y1 = data1['Class']
    Y2 = data2['Class']
    xtrain1 = X1.iloc[:30]
    xtest1 = X1.iloc[30:]
    xtrain2 = X2.iloc[:30]
    xtest2 = X2.iloc[30:]
    ytrain1 = Y1.iloc[:30]
    ytest1 = Y1.iloc[30:]
    ytrain2 = Y2.iloc[:30]
    ytest2 = Y2.iloc[30:]
    xtr = pd.concat([xtrain1, xtrain2])
    xtst = pd.concat([xtest1, xtest2])
    ytr = pd.concat([ytrain1, ytrain2])
    ytst = pd.concat([ytest1, ytest2])
    labels1 = ytr.unique()
    labels2 = ytst.unique()
    ytr = ytr.replace({labels1[0]: 1, labels1[1]: -1})
    ytst = ytst.replace({labels2[0]: 1, labels2[1]: -1})
    xtr = np.array(xtr)
    xtst = np.array(xtst)
    if bias == 'Yes':
        xtr = np.insert(xtr, 0, np.ones([1, 1]), axis=1)
        xtst = np.insert(xtst, 0, np.ones([1, 1]), axis=1)
    ypred = np.array(int)

    # Training Start
    W = intialize(bias)
    xtr = np.array(xtr)
    ytr = np.array(ytr)
    W = nn_model_adaline(xtr, ytr, W, epochs, l_rate, threshold)

    # Testing on Weights Retrieved from NN_Model
    xtst = np.array(xtst)
    for row in range(xtst.shape[0]):
        Z, A = forward(np.expand_dims(xtst[row], axis=1), W)
        ypred = np.append(ypred, A)
    error = 0
    inCorrectC1 = 0
    inCorrectC2 = 0
    ytst = np.array(ytst)
    for i in range(ytst.shape[0]):
        if ytst[i] != ypred[i]:
            error += 1
            if ytst[i] == 1:
                inCorrectC1 += 1
            else:
                inCorrectC2 += 1
    correctC1 = 20 - inCorrectC1
    correctC2 = 20 - inCorrectC2
    confusionMatrix = [[correctC1, inCorrectC1], [correctC2, inCorrectC2]]
    accuracy = ((correctC1 + correctC2) / 40) * 100

    # Confusion Matrix and Accuracy
    print(confusionMatrix)
    print("Error = ", error)
    print("Overall Accuracy = " + str(accuracy) + "%")
    return W, bias


# Neural Network Model Algorithm Adaline
def nn_model_adaline(X, Y, W, num_of_iterations, learning_rate, threshold):
    for i in range(num_of_iterations):
        prediction = np.empty(Y.shape[0])
        for row in range(X.shape[0]):
            Z, A = forward(np.expand_dims(X[row], axis=1), W)
            prediction[i] = A
            if A != Y[row]:
                error = Y[row] - A
                W = W + learning_rate * error * np.expand_dims(X[row], axis=1)
        MSE = ((1 / 2) * X.shape[0]) * ((np.sum(Y - prediction)) ** 2)
        if MSE < threshold:
            break
    return W


# Back Propagation Model
def backPropagation_algo(options_list_features, options_list_classes, l_rate, epochs, bias,
                         hidden_layers, neurons, choosenFunction):
    # Write Code Here
