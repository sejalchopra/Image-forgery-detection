import matplotlib.pyplot as plt
import pandas as pd


def plot_epochs(metric1, metric2, ylab):
    """
    Plot the metrics of both datasets
    :param metric1: The first metric that we want to plot
    :param metric2: The second metric that we want to plot
    :param ylab: The label of the y axis
    """
    plt.plot(metric2, label='CASIA2')
    plt.plot(metric1, label='NC16')
    plt.ylabel(ylab)
    plt.xlabel("Epoch")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    df1 = pd.read_csv(filepath_or_buffer="../../data/output/accuracy/CASIA2_Accuracy.csv")
    df2 = pd.read_csv(filepath_or_buffer="../../data/output/accuracy/NC16_Accuracy.csv")
    df3 = pd.read_csv(filepath_or_buffer="../../data/output/loss_function/CASIA2_Loss.csv")
    df4 = pd.read_csv(filepath_or_buffer="../../data/output/loss_function/NC16_Loss.csv")
    plot_epochs(df1.iloc[:, 1], df2.iloc[:, 1], 'Training Accuracy')
    plot_epochs(df3.iloc[:, 1], df4.iloc[:, 1], 'Training Loss')
