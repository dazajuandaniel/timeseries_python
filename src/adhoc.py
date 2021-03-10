"""
Python files for one-off functions
"""
import pandas as pd
import numpy as np

def adhoc_read_data():
    """
    Function that given two files returns the merge of both
    """
    file1 = pd.read_csv('../data/01_raw_data/File 1.csv')
    file2 = pd.read_csv('../data/01_raw_data/File 2.csv')

    # Parse Dates to Datetime
    file1['Time'] = pd.to_datetime(file1.Time, format='%d/%m/%y',errors = 'raise')
    file2['Time'] = pd.to_datetime(file2.Time, format='%d/%m/%y',errors = 'raise')

    # Merge files
    all_files = pd.concat([file1,file2])
    all_files = all_files.sort_values(by = 'Time', ascending=True).reset_index(drop=True)
    all_files = all_files.reset_index(drop=False)
    all_files = all_files.rename(columns={'index':'item_id'})

    return all_files


def adhoc_merge_columns(df):
    """
    Function that merges the required columns
    """
    cols = [x for x in df.columns if 'CUB' in x and 'Share' in x]
    cols_comp = [x for x in df.columns if 'Comp' in x and 'Share' in x]
    df['CUB.Share.top.8'] = df[cols].sum(axis = 1)
    df['Comp.Share.top.8'] = df[cols_comp].sum(axis = 1)

    return df

def load_data(path):
    """
    Function that loads the CSV file and sets the index
    Args:
        path(str): Path to data
    Returns:
        data(df): Dataframe with loaded data
    """
    # Load Data
    data = pd.read_csv(path)

    # Set Index in the correct format
    data = data.set_index('date_range')
    index_ = pd.to_datetime(data.index,infer_datetime_format=True)
    date_index = pd.DatetimeIndex(index_,freq='W-SUN')
    
    # Change data's index
    data = data.set_index(date_index)
    
    return data