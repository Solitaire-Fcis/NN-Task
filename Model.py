import matplotlib.pyplot as plt
import pandas as pd


def read_data(path):
    content = pd.read_csv(path)
    content.to_csv()
    return content


def plot_data(dataset, feature1, feature2):
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
    plt.legend()
    plt.show()
