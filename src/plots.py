
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List

class PlotInterface:
    """Interface for plot classes that can be plotted on a matplotlib axis, wrapped by a PlotGrid."""

    def plot(self, ax=None):
        """Plot the visualization on the given axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates a new figure.
        """
        pass

class PlotGrid:
    """A grid layout for plotting multiple plots in a single figure."""

    def __init__(self, n_cols, cell_size=(5, 4)):
        """Initialize the plot grid.

        Args:
            n_cols (int): Number of columns in the grid.
            cell_size (tuple): Size of each cell (width, height).
        """
        self.n_cols = n_cols
        self.cell_size = cell_size
        self.fig = None
        self.clear()
    
    def create_grid(self, num_plots):
        """Create the grid of subplots.

        Args:
            num_plots (int): Number of plots to accommodate.
        """
        self.n_rows = math.ceil(num_plots / self.n_cols)
        figsize = (self.cell_size[0]*self.n_cols, self.cell_size[1]*self.n_rows)
        self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, figsize=figsize)
        self.axes = self.axes.flatten()
        self.current_index = 0
    
    def __get_next_axis(self):
        """Get the next available axis in the grid.

        Returns:
            matplotlib.axes.Axes: The next axis.

        Raises:
            ValueError: If grid not created.
            IndexError: If no more axes available.
        """
        if self.fig is None or self.axes is None:
            raise ValueError("Grid not created. Call create_grid() first.")
        elif self.current_index >= len(self.axes):
            raise IndexError("No more axes available in the grid.")
        else:
            ax = self.axes[self.current_index]
            self.current_index += 1
            return ax
    
    def plot(self, plot_obj: PlotInterface):
        """Plot a single plot object on the next axis.

        Args:
            plot_obj (PlotInterface): The plot object to plot.
        """
        ax = self.__get_next_axis()
        plot_obj.plot(ax)

    def plot_all(self, plot_objects: List[PlotInterface]):
        """Plot all plot objects in the grid.

        Args:
            plot_objects (List[PlotInterface]): List of plot objects to plot.
        """
        self.create_grid(len(plot_objects))
        for plot_obj in plot_objects:
            self.plot(plot_obj)
        self.show()
    
    def show(self):
        """Display the grid of plots."""
        if self.fig is None:
            raise ValueError("Grid not created. Call create_grid() first.")
        plt.tight_layout()
        plt.show()
    
    def clear(self):
        """Clear the current grid and close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
            self.current_index = 0
    
class Histogram(PlotInterface):
    """A histogram plot with mean, median, mode lines and normal curve overlay."""

    def __init__(self, data, title, x_label=""):
        """Initialize the histogram.

        Args:
            data (array-like): The data to plot.
            title (str): The title of the plot.
            x_label (str): The label for the x-axis.
        """
        self.data = data
        self.title = title
        self.xlabel = x_label
    
    def plot(self, ax = None):
        """Plot the histogram on the given axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates a new figure.
        """
        num_bins = 30
        mean_val = self.data.mean()
        median_val = self.data.median()
        modes = self.data.mode()
        mode_val = modes if not modes.empty else []

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        # Plot histogram
        ax.hist(self.data, bins=num_bins, alpha=0.7, density=True) # Use density=True for normal curve overlay
        ax.set_title(self.title, fontsize=11)
        ax.set_xlabel(self.xlabel, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)

        # Plot mean, median, mode lines
        max_modes = 7
        for mode in mode_val:
            ax.axvline(mode, color='lightgray', linestyle='dashed', linewidth=2, label=f'Mode: {mode:.2f}')
            max_modes -= 1
            if max_modes == 0:
                break
        
        if mean_val is not None:
            ax.axvline(mean_val, color='red', linestyle='solid', linewidth=2, label=f'Mean: {mean_val:.2f}')
        if median_val is not None:
            ax.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')

        # Plot approximated normal curve
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, num_bins)
        p = norm.pdf(x, mean_val, self.data.std())
        ax.plot(x, p, 'k', linewidth=2, label='Normal Curve')

        ax.legend()
        if fig is not None:
            plt.show()

class CorrelationHeatmap(PlotInterface):
    """A correlation heatmap plot using seaborn."""

    def __init__(self, df, columns, title='Correlation Heatmap'):
        """Initialize the correlation heatmap.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            columns (list): List of columns to include in the correlation.
            title (str): The title of the plot.
        """
        self.df = df
        self.columns = columns
        self.title = title

    def plot(self, ax = None):
        """Plot the correlation heatmap on the given axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates a new figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 12))

        df_corr = self.df[self.columns].dropna()
        # Calculate the correlation matrix
        corr_matrix = df_corr.corr()

        # Plotting the heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title(self.title)

        if fig is not None:
            plt.show()

class FrequencyPie(PlotInterface):
    """A pie chart showing frequency distribution of categorical data."""

    def __init__(self, data, title, labels:dict=None, figsize=(5, 4)):
        """Initialize the frequency pie chart.

        Args:
            data (array-like): The categorical data to plot.
            title (str): The title of the plot.
            labels (dict, optional): Mapping of values to labels.
            figsize (tuple): Size of the figure.
        """
        self.data = data
        self.title = title
        self.labels = labels
        self.figsize = figsize

    def plot(self, ax = None):
        """Plot the pie chart on the given axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates a new figure.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        counts = self.data.value_counts()
        sizes = counts.values
        if self.labels is not None:
            labels = [self.labels.get(val, str(val)) for val in counts.index]
        else:
            labels = counts.index.astype(str)

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title(self.title)

        if fig is not None:
            plt.show()

class ScatterPlot(PlotInterface):
    """A scatter plot with mean and median horizontal lines."""

    def __init__(self, x, y, title, x_label="", y_label="", x_unit="", y_unit=""):
        """Initialize the scatter plot.

        Args:
            x (array-like): X-axis data.
            y (array-like): Y-axis data.
            title (str): The title of the plot.
            x_label (str): Label for x-axis.
            y_label (str): Label for y-axis.
            x_unit (str): Unit for x-axis.
            y_unit (str): Unit for y-axis.
        """
        self.x = x
        self.y = y
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.x_unit = x_unit
        self.y_unit = y_unit

    def plot(self, ax=None):
        """Plot the scatter plot on the given axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates a new figure.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        median = self.y.median()
        mean = self.y.mean()
        ax.scatter(self.x, self.y, alpha=0.5)
        ax.axhline(y=mean, color='r', linestyle='-', label=f"Mean: {mean:.2f}")
        ax.axhline(y=median, color='y', linestyle='--', label=f"Median: {median:.2f}")
        ax.set_title(self.title)
        ax.set_xlabel(f'{self.x_label} ({self.x_unit})')
        ax.set_ylabel(f'{self.y_label} ({self.y_unit})')
        ax.legend()

        if fig is not None:
            plt.show()

class CrossTabBar(PlotInterface):
    """A stacked bar chart for cross-tabulated data."""

    def __init__(self, x_data, y_data, title, x_label="", y_label="", x_value_labels:dict=None, y_value_labels:dict=None, legend_title=None):
        """Initialize the cross-tab bar chart.

        Args:
            x_data (array-like): Data for x-axis.
            y_data (array-like): Data for y-axis.
            title (str): The title of the plot.
            x_label (str): Label for x-axis.
            y_label (str): Label for y-axis.
            y_col (str): Column for y-axis.
            title (str): The title of the plot.
            x_label (str): Label for x-axis.
            y_label (str): Label for y-axis.
            x_value_labels (dict, optional): Mapping for x-axis values.
            y_value_labels (dict, optional): Mapping for y-axis values.
            legend_title (str, optional): Title for the legend.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.x_value_labels = x_value_labels
        self.y_value_labels = y_value_labels
        self.legend_title = legend_title if legend_title else y_col
    
    def plot(self, ax=None):
        """Plot the stacked bar chart on the given axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates a new figure.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        crosstab = pd.crosstab(self.x_data, self.y_data, normalize='index') * 100
        crosstab.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_xticklabels([self.x_value_labels.get(val, str(val)) for val in crosstab.index], rotation=0)
        ax.legend(title=self.legend_title, labels=[self.y_value_labels.get(val, str(val)) for val in crosstab.columns])

        # Add percentage labels on the bars
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.text(x+width/2, y+height/2, f'{height:.1f}%', ha='center', va='center')

        if fig is not None:
            plt.show()

class SplitViolin(PlotInterface):
    """A split violin plot comparing distributions of a value across two categories."""

    def __init__(self, y_data, split_data, title, x_label="", y_label="", split_labels:dict=None):
        """Initialize the split violin plot.

        Args:
            y_data (array-like): The data for the values to compare.
            split_data (array-like): The data for the categories to split by.
            title (str): The title of the plot.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            split_labels (dict, optional): Mapping for split categories to labels.
        """
        self.y_data = y_data
        self.split_data = split_data
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.labels = split_labels if split_labels else {}
    
    def plot(self, ax=None):
        """Plot the split violin plot on the given axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates a new figure.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        data = pd.DataFrame({
            self.y_label: self.y_data, 
            'split': self.split_data,
            'x': '' # Add a dummy x column for seaborn's violinplot when splitting by hue
        })
        sns.violinplot(
            data=data, 
            y=self.y_label, 
            x='x', 
            split=True, 
            inner='box',
            hue='split',
            ax=ax
        )
        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        handles, labels = ax.get_legend_handles_labels()
        labels = [self.labels.get(val, str(val)) for val in self.labels.keys()]
        ax.legend(handles, labels, title=None)

        if fig is not None:
            plt.show()

class TimeSeriesPlot(PlotInterface):
    """A line plot for time series data."""

    class Line:
        """A single line in the time series plot."""
        def __init__(self, value_column, label, color, linestyle='-'):
            self.value_column = value_column
            self.label = label
            self.color = color
            self.linestyle = linestyle


    def __init__(self, df, date_column, lines: List[Line], title=None):
        """Initialize the time series plot.

        Args:
            df (pandas.DataFrame): The dataframe containing the time series data.
            date_column (str): The name of the column containing the dates.
            lines (list): A list of Line objects containing the y-axis data and metadata.
            title (str): The title of the plot.
        """
        self.df = df
        self.date_column = date_column
        self.lines = lines
        self.title = title

    def plot(self):
        """Plot the time series line on the given axis.
        """
        plt.figure(figsize=(15, 6))

        for line in self.lines:
             sns.lineplot(
                data=self.df, 
                x=self.date_column, 
                y=line.value_column, 
                label=line.label, 
                color=line.color, 
                linestyle=line.linestyle, 
            )
        plt.title(self.title)
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
