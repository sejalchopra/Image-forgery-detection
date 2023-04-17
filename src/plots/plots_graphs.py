import matplotlib.pyplot as plt
import pandas as pd


def plot_epochs(metric1, metric2, ylab):
    """
    Plots the training accuracy and training loss for CASIA2 and NC16 datasets.

    Parameters:
    metric1 (Pandas Series): Training metric for dataset 1.
    metric2 (Pandas Series): Training metric for dataset 2.
    ylab (str): Label for y-axis of the plot.

    Returns:
    None
    """
    plt.plot(metric1, label='CASIA2')
    plt.plot(metric2, label='NC16')
    plt.ylabel(ylab)
    plt.xlabel("Epoch")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    # Reading CSV files
    df1 = pd.read_csv(filepath_or_buffer="../../data/output/accuracy/CASIA2_Accuracy.csv")
    df2 = pd.read_csv(filepath_or_buffer="../../data/output/accuracy/NC16_Accuracy.csv")
    df3 = pd.read_csv(filepath_or_buffer="../../data/output/loss_function/CASIA2_Loss.csv")
    df4 = pd.read_csv(filepath_or_buffer="../../data/output/loss_function/NC16_Loss.csv")

    # Plotting training accuracy and loss
    plot_epochs(df1.iloc[:, 1], df2.iloc[:, 1], 'Training Accuracy')
    plot_epochs(df3.iloc[:, 1], df4.iloc[:, 1], 'Training Loss')

