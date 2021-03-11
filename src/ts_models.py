"""
Python Script used for various forecasting techniques
"""
# Data Processing
import pandas as pd
import numpy as np
from itertools import product
import math

# Model
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


# Local Files
import sys
sys.path.append("../src")
import ts_plots, ts_preprocess

def calculate_decompose(df,**kwargs):
    """
    Wrapper function to calculate the decomposition using statsmodel
    Args:
        df(pd.DataFrame): The timeseries dataframe indexed with date
    Returns:
        decompose(DecomposeResult): Resulting values
    """
    model = kwargs.get('model','additive')

    return sm.tsa.seasonal_decompose(df, model=model)

def adf_test(ts, threshold):
    """
    Function that returns the Augmented Dickey-Fuller Test
    Args:
        ts(pd.Series): The timeseries to be tested
    Returns:
        metrics(dict): Dictionary with the results of the test
    Pending:
        Convert to Object for reproducibility
    """
    metrics = {}
    aftest = adfuller(ts, autolag='AIC')
    metrics['statistic'] = aftest[0]
    metrics['pvalue'] = aftest[1]
    metrics['critical_values'] = aftest[4]
    
    metrics['result'] = 'not_stationary' if metrics['critical_values'][threshold] < metrics['statistic'] else 'stationary'
    
    return metrics

def kpss_test(ts,threshold):
    """
    Function that returns the Kwiatkowski-Phillips-Schmidt-Shin Test
    Args:
        ts(pd.Series): The timeseries to be tested
    Returns:
        metrics(dict): Dictionary with the results of the test
    Pending:
        Convert to Object for reproducibility
    """
    metrics = {}
    ksstest = kpss(ts)
    metrics['statistic'] = ksstest[0]
    metrics['pvalue'] = ksstest[1]
    metrics['critical_values'] = ksstest[3]
    
    metrics['result'] = 'not_stationary' if metrics['critical_values'][threshold] < metrics['statistic'] else 'stationary'
        
    
    return metrics

def check_stationary(ts, **kwargs):
    """
    Function that automatically checks for Stationary in the data.
    Performs the following tests:
        * Augmented Dickey-Fuller Test
        * KPSS
    Args:
        ts(pd.Series): The timeseries that is going to be analyzed
        kwargs:
            show_plot (bool): Boolean flag indicating if the Rolling statistics
                              plots should be created
            window (int): The rolling window for the statistics. Defaults to 12
            verbose (bool): Flag to indicate if the results should be logged
    Returns:
        metrics (dict): Dictionary with the results of the tests
    """
    # Initialise Metrics Dictionary
    metrics = {}
    
    # Try kwargs
    threshold = kwargs.get('threshold','1%')
    if threshold is None:
        threshold = '1%'
        
    show_plot = kwargs.get('show_plot',False)
    window = kwargs.get('window',12)
    verbose = kwargs.get('verbose',False)
    
    if show_plot is not None and window is None:
        window = 12
    
    # Run Augmented Dickey-Fuller Test
    metrics['ADF_Test'] = adf_test(ts, threshold)
    metrics['KSS_Test'] = kpss_test(ts, threshold)
    
    if show_plot:
        fig = ts_plots.plot_rolling(ts,window)
        fig.show()
    
    return metrics

def exponential_smoothing(ts_train,ts_test, **kwargs):
    """
    Function that gets the Exponential Smoothing values
    Args:
        ts_train
        ts_test
    Return:
        data
        
    """
    
    # Get Default Values
    seasonal_periods = kwargs.get('seasonal_periods',None)
    seasonal = kwargs.get('seasonal',None)
    smoothing_level = kwargs.get('smoothing_level',None)
    use_boxcox = kwargs.get('use_boxcox',None)
    damped = kwargs.get('damped',False)
    freq = kwargs.get('freq',False)
    trend = kwargs.get('trend',None)
    remove_bias = kwargs.get('remove_bias',True)
    return_model = kwargs.get('return_model',False)
    damping_trend = kwargs.get('damping_trend',None)

    # Get Param Values
    model_train = kwargs.get('model_train',None)

    # Fit Model
    if model_train is None:
        model = ExponentialSmoothing(ts_train, trend = trend, damped = damped,
                                    seasonal = seasonal, seasonal_periods = seasonal_periods)\
                .fit(smoothing_level=smoothing_level, use_boxcox=use_boxcox, remove_bias=remove_bias,damping_trend=damping_trend)
    else:
        model_params = model_train.params
        model = ExponentialSmoothing(ts_train, trend = trend, damped = damped,
                                    seasonal = seasonal, seasonal_periods = seasonal_periods)\
            .fit(smoothing_level=model_params['smoothing_level'], use_boxcox=model_params['use_boxcox'], 
                 remove_bias=model_params['remove_bias'], initial_trend = model_params['initial_trend'],
                 smoothing_seasonal= model_params['smoothing_seasonal'], damping_trend = model_params['damping_trend'],
                 initial_level = model_params['initial_level'])
    
    fcast1 = model.predict(start=ts_test.index[0], end=ts_test.index[-1])
    
    if return_model:
        return get_performance_metrics(ts_test,fcast1), model
    else:
        return get_performance_metrics(ts_test,fcast1)

def get_performance_metrics(y_true, y_pred, include='all'):
    """
    Function that automatically runs and gets multiple performance metrics
    """
    res_dic = {}
    metric_dict = {'RMSE':mean_squared_error,
                   'MAE':mean_absolute_error,
                   'MAPE':mean_absolute_percentage_error,
                   'MASE':mean_absolute_error}
    
    for i in metric_dict.keys():
        res_dic[i] = round(metric_dict[i](y_true,y_pred),2)
        
    return res_dic

def aic(observations,variables,pred_vector,true_vector):
    """
    Function that calculates the AIC value for a dataframe
    Args:
        observations
        variables
        pred_vector
        true_vector
    Returns:
        aic
    """
    return math.log(sum((pred_vector-true_vector)**2)/observations)*observations+ 2*(variables+2)

def bic(observations,variables,pred_vector,true_vector):
    """
    Function that calculates the BIC value for a dataframe
    Args:
        observations
        variables
        pred_vector
        true_vector
    Returns:
        bic
    """
    return math.log(sum((pred_vector-true_vector)**2)/observations)*observations+ (variables+2)*math.log(observations)

def search_best_combination(y,X, sort_by='AIC'):
    """
    Function that iterates through all combinations and calculates a regression each time
    Each combination has its own performance metric results
    Args:
        y: Dependent variable
        X: Feature values
    Returns:
        df_res: Dataframe with recorded performance metrics
    """
    res_dic = {}
    df_res = pd.DataFrame()
    col_dict = {}
    
    cols_dict,binary_values = ts_preprocess.generate_combinations(X.columns)
    
    for index,x in enumerate(cols_dict.keys()):
        
        # Get Columns Iteration
        x_col_sel = list(cols_dict[x])

        # Store Current Col Iteration
        col_dict[index] = x_col_sel
        
        # Subset Data
        data_x = X[x_col_sel]
        
        # Train Model
        model = sm.OLS(endog = y,exog= add_constant(data_x)).fit()
        
        # Get Metrics
        res_dic['AIC'] = aic(X.shape[0],len(x_col_sel),y,model.predict())
        res_dic['BIC'] = bic(X.shape[0],len(x_col_sel),y,model.predict())
        res_dic['AdjR2'] = model.rsquared_adj
        res_dic['combination'] = str(binary_values[x])
        
        # Build DataFrame
        df_res = pd.concat([df_res,pd.DataFrame(res_dic,index=[index])])
        
    return df_res.sort_values(by = sort_by)