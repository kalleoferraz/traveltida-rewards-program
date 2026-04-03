
import pandas as pd
import numpy as np
from src.plots import PlotGrid, KDEComparisonPlot


class Profiler:
    """Class to identify the profile of a segment by comparing it to the remaining population using Cohen's d effect size and KDE visualizations."""

    def __init__(self, df, segment_condition, min_effect_size=0.5):
        """Initialize the Profiler with the segment and remaining dataframes, and the minimum effect size threshold.
        
        Args:
            df (pd.DataFrame): The full dataset to profile.
            segment_condition (pd.Series): Boolean series indicating which rows belong to the segment.
            min_effect_size (float): Minimum Cohen's d effect size threshold for inclusion in profile. Defaults to 0.5.
        """
        self.df = df
        self.condition = segment_condition
        self.min_effect_size = min_effect_size
        self.profile = None
    
    def get_profile(self):
        """Calculate Cohen's d effect size for each column and store in the profile."""
        self.profile = {}
        df_segment = self.df[self.condition]
        df_remaining = self.df.loc[~self.condition]
        # Calculate Cohen's d for each numeric columns
        for column in df_segment.columns:
            if pd.api.types.is_numeric_dtype(df_segment[column]):
                mean_diff = df_segment[column].mean() - df_remaining[column].mean()
                pooled_std = np.sqrt((df_segment[column].std() ** 2 + df_remaining[column].std() ** 2) / 2)
                cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
                # Save cohen's d and means if it meets the effect size threshold
                if abs(cohen_d) >= self.min_effect_size:
                    self.profile[column] = {
                        'cohen_d': cohen_d,
                        'mean_segment': df_segment[column].mean(),
                        'mean_remaining': df_remaining[column].mean(),
                        'lift': (df_segment[column].mean() / df_remaining[column].mean() - 1) if df_remaining[column].mean() != 0 else float('inf'),
                    }
        # Sort profile by absolute effect size in descending order
        self.profile = dict(sorted(self.profile.items(), key=lambda item: abs(item[1]['cohen_d']), reverse=True))
        return self.profile

    def to_dataframe(self):
        """Convert the profile dictionary to a DataFrame for easier analysis."""
        if self.profile is None:
            self.get_profile()
        return pd.DataFrame.from_dict(self.profile, orient='index')

    def plot_profile(self):
        """Plot KDE comparisons for the columns that meet the effect size threshold."""
        if self.profile is None:
            self.get_profile()
        if not self.profile:
            print("No columns meet the effect size threshold for the given segment.")
            return
        
        df_segment = self.df[self.condition]
        df_remaining = self.df.loc[~self.condition]
        PlotGrid(n_cols=min(3, len(self.profile))).plot_all([
            KDEComparisonPlot(df_segment[column], df_remaining[column], 
                title = f"{column} (lift={self.profile[column]['lift']:.2%}, d={self.profile[column]['cohen_d']:.2f})")
            for column in self.profile.keys() 
        ])