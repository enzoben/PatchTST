import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def forecasting(model: nn.Module, set: torch.utils.data.Dataset, item: int, device: torch.device = "cpu"):
    """forcast the the item from the set using the model

    Args:
        model (nn.Module): the model to use
        set (torch.utils.data.Dataset): the set used to forcast the data
        item (int): index of the item to forcast
        device (torch.device, optional): Defaults to "cpu".
    """
    model.to(device)
    model.eval()
    x, y, x_time, y_time = set.__getitem__(item)
    x, y = x.to(device), y.to(device)

    x_time = pd.DataFrame({"date" : pd.to_datetime(x_time)})
    y_time = pd.DataFrame({"date" : pd.to_datetime(y_time)})

    y_hat = model(x.unsqueeze(0).permute(0,2,1))["prediction_outputs"].squeeze(0)

    columns = set.data.columns

    x_df = pd.DataFrame(x.permute(1,0).cpu().detach().numpy(), columns=columns)
    x_df["date"] = x_time["date"]
    x_df = x_df.set_index("date")

    y_df_true = pd.DataFrame(y.permute(1,0).cpu().detach().numpy(), columns=columns)
    y_df_true["date"] = y_time["date"]
    y_df_true = y_df_true.set_index("date")

    y_df_hat = pd.DataFrame(y_hat.cpu().detach().numpy(), columns=columns)
    y_df_hat["date"] = y_time["date"]
    y_df_hat = y_df_hat.set_index("date")


    return y_df_hat, y_df_true, x_df

def compute_mse(model: nn.Module, test_set: torch.utils.data.Dataset, device: torch.device = "cpu"):
    """This function allow us to compute the MSE of the model

    Args:
        model (nn.Module): the model to use
        test_set (torch.utils.data.Dataset): the test set

    Returns:
        float: the MSE of the model
    """

    criterion = nn.MSELoss()
    loss = []

    dloader = tqdm(DataLoader(test_set, batch_size=1, shuffle=True), unit="batches")

    model.to(device)
    model.eval()

    for i, (x, y, _, _) in enumerate(dloader, start = 1):
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            y_hat = model(x.permute(0,2,1))["prediction_outputs"]
            loss.append(criterion(y_hat, y.permute(0,2,1)).item()/y_hat.shape[2])
            dloader.set_description("TEST : loss : {:.3f} (+- {:.3f}) ".format(np.mean(loss), np.std(loss)))

def plot_forecasting(model: torch.nn.Module, 
                     dataset: torch.utils.data.Dataset, 
                     item: int, 
                     device: torch.device = "cpu",
                     columns_to_plot: list = None):
    """
    Plots the forecasting results for a specific sample in the dataset.

    Args:
        model (torch.nn.Module): The trained forecasting model.
        dataset (torch.utils.data.Dataset): The dataset containing the time series.
        item (int): The index of the sample to visualize.
        device (torch.device, optional): Device to use for inference. Defaults to "cpu".
        columns_to_plot (list, optional): List of column names to plot in separate subplots.
                                          If None, only the first column is plotted.
    """
    
    # Extract x_df (input), y_hat (forecasted values), and y_true (ground truth)
    y_hat, y_true, x_df = forecasting(model, dataset, item, device)

    # Ensure the data is a Pandas DataFrame
    y_hat, y_true, x_df = map(pd.DataFrame, [y_hat, y_true, x_df])

    # Get available columns
    available_columns = x_df.columns.tolist()

    # Default: Plot only the first column if no list is provided
    if columns_to_plot is None:
        columns_to_plot = [available_columns[0]]

    # Number of subplots
    num_plots = len(columns_to_plot)

    # Create the main figure
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(12, 4 * num_plots), sharex=True)
    sns.set_style("darkgrid")

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot case

    # Dictionary to store legend handles
    legend_handles = []

    # Iterate over each selected column
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        color = sns.color_palette("husl", num_plots)[i]  # Generate unique colors
        
        # Extract time indices
        time_index_x = x_df.index
        time_index_y = y_true.index

        # Background shading for input range (darker blue)
        ax.axvspan(time_index_x[0], time_index_x[-1], facecolor="blue", alpha=0.2, label="Input Region")

        # Background shading for forecast range (lighter blue)
        ax.axvspan(time_index_y[0], time_index_y[-1], facecolor="lightblue", alpha=0.2, label="Forecast Region")

        # Plot input (x_df)
        line_x, = ax.plot(time_index_x, x_df[col], color="blue", linestyle="-", label=f"Historical data", alpha=0.9)

        # Plot forecast (y_hat)
        line_hat, = ax.plot(time_index_y, y_hat[col], color="red", linestyle="-", label=f"Forecasting", alpha=0.9)

        # Plot ground truth (y_true)
        line_true, = ax.plot(time_index_y, y_true[col], color="green", linestyle="dotted", label=f"Ground Truth", alpha=0.8)

        # Store unique legend handles (only once)
        if i == 0:
            legend_handles.extend([line_x, line_hat, line_true])

        # Y-axis label
        ax.set_ylabel(f"Value ({col})", fontsize=12)

    # X-axis label (only for last subplot)
    axes[-1].set_xlabel("Time", fontsize=12)

    # Single legend for all subplots
    fig.legend(handles=legend_handles, loc="upper right", fontsize=12, title="Legend", frameon=True)

    plt.xlim(time_index_x[0], time_index_y[-1])

    # Adjust layout
    plt.suptitle(f"Forecasting Visualization for Sample {item}/{dataset.__len__()}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the legend and title
    plt.grid(True)

    # Show the plot
    plt.show()