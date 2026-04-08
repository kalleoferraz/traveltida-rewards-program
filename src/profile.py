
import pandas as pd
import numpy as np
from src.plots import PlotGrid, KDEComparisonPlot


class Profiler:
    """Class to identify the profile of a segment by comparing it to the remaining population using Cohen's d effect size and KDE visualizations."""

    def __init__(self, df_segment, df_remaining, min_effect_size=0.5):
        """Initialize the Profiler with the segment and remaining dataframes, and the minimum effect size threshold.
        
        Args:
            df_segment (pd.DataFrame): The dataframe for the segment to profile.
            df_remaining (pd.DataFrame): The dataframe for the remaining population.
            min_effect_size (float): Minimum Cohen's d effect size threshold for inclusion in profile. Defaults to 0.5.
        """
        self.df_segment = df_segment
        self.df_remaining = df_remaining
        self.min_effect_size = min_effect_size
        self.profile = None
    
    def get_profile(self):
        """Calculate Cohen's d effect size for each column and store in the profile."""
        self.profile = {}
        # Calculate Cohen's d for each numeric columns
        for column in self.df_segment.columns:
            if pd.api.types.is_numeric_dtype(self.df_segment[column]):
                mean_diff = self.df_segment[column].mean() - self.df_remaining[column].mean()
                pooled_std = np.sqrt((self.df_segment[column].std() ** 2 + self.df_remaining[column].std() ** 2) / 2)
                cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
                # Save cohen's d and means if it meets the effect size threshold
                if abs(cohen_d) >= self.min_effect_size:
                    self.profile[column] = {
                        'cohen_d': cohen_d,
                        'mean_segment': self.df_segment[column].mean(),
                        'mean_remaining': self.df_remaining[column].mean(),
                        'lift': (self.df_segment[column].mean() / self.df_remaining[column].mean() - 1) if self.df_remaining[column].mean() != 0 else float('inf'),
                    }
        # Sort profile by absolute effect size in descending order
        self.profile = dict(sorted(self.profile.items(), key=lambda item: abs(item[1]['cohen_d']), reverse=True))
        return self.profile

    def to_dataframe(self):
        """Convert the profile dictionary to a DataFrame for easier analysis."""
        if self.profile is None:
            self.get_profile()
        return pd.DataFrame.from_dict(self.profile, orient='index')

    def plot_profile(self, n_cols=4):
        """Plot KDE comparisons for the columns that meet the effect size threshold."""
        if self.profile is None:
            self.get_profile()
        if not self.profile:
            print(f"No significant difference from population found. (No column meet the effect size threshold ({self.min_effect_size}) for the given segment)")
            return
        
        PlotGrid(n_cols=min(n_cols, len(self.profile))).plot_all([
            KDEComparisonPlot(self.df_segment[column], self.df_remaining[column], 
                title = f"{column} (lift={self.profile[column]['lift']:.2%}, d={self.profile[column]['cohen_d']:.2f})")
            for column in self.profile.keys() 
        ])