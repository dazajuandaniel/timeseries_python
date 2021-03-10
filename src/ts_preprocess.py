"""
Python Script used to Data Processing
"""
# Data Manipulation
import pandas as pd
import numpy as np

# Data Visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ipywidgets import widgets, interact, interact_manual
from IPython.display import display, clear_output
from itertools import product
import functools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Timeseries Manipulation
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

# Other
import warnings

def create_timeseries_series(df, cols, frequency):
    """
    Function that creates a series from a dataframe which is the Timeseries
    Args:
        df
    Returns:
        ts
    """
    ts_cols = cols
    ts_df = df[ts_cols].copy()
    ts_df = ts_df.set_index(ts_cols[0])

    ts_df = ts_df[ts_cols[1]].resample(frequency).mean()
    all_dates = pd.date_range(start = ts_df.index[0],end = ts_df.index[-1], freq="W",name='date_range')

    ts_df = ts_df.reindex(all_dates)
    validate_data(ts_df)
    return ts_df

def validate_data(df):
    """
    Function to validate datasets
    """
    # Check for null values
    if df.isna().sum() != 0:
        raise Exception("DataFrame has NULL values, please validate")
    return

def validate_datetime_index(df, allow = True):
    """
    Function that validates if a dataframe has a datetime index
    Args:
        df(pd.DataFrame): The timeseries dataframe indexed with date
    Returns:
        check: Boolean value that determines if the index is datetime
    Raises:
        Exception
    """
    if not isinstance(df.index,pd.DatetimeIndex) and allow is False:
        raise Exception("DataFrame is not indexed with a datetime, please index and try again")
    elif not isinstance(df.index,pd.DatetimeIndex) and allow:
        warnings.warn("Warning DataFrame is not indexed with a datetime")
        return False
    else:
        return True

def combinations_on_off(num_classifiers):
    """
    Function that calculates the on/off combinations for the total number of variables
    Args:
        num_classifiers
    Returns:
        array
    """
    return [[int(x) for x in list("{0:0b}".format(i).zfill(num_classifiers))]
            for i in range(1, 2 ** num_classifiers)]

def generate_combinations(variables):
    """
    Function that calculates the total combinations for a given array
    Args:
        variables
    Returns:
        cols_dict: Dictionary with possible combinations
        binary_values: The values in a binary array
    """
    cols_dict = {}
    binary_values = {}
    for row,value in enumerate(combinations_on_off(len(variables))):
        binary_values[row] = value
        cols_dict[row] = []
        for index,switch_value in enumerate(value):
            if switch_value == 1:
                cols_dict[row].append(variables[index])
                
    return cols_dict, binary_values