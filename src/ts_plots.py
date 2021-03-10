"""
Python Script used for plotting common timeseries charts
"""
# Data Manipulation
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import shift

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
import sys
import warnings
sys.path.append("../src")
import ts_preprocess, ts_models

def plot_rolling(ts, window):
    """
    Function that plots the rolling window value for the tiemseries
    Args:
        ts(pd.Series): The timeseries to be tested
    Returns:
        fig(plotly.fig): The figure value
    """
    #Determing rolling statistics
    x = ts.index
    rolmean = pd.Series(ts).rolling(window=window).mean().values 
    rolstd = pd.Series(ts).rolling(window=window).std().values  

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=ts.values ,name = 'original_values'))

    fig.add_trace(
        go.Scatter(x=x, y=rolmean ,name = 'rolling_mean'))

    fig.add_trace(
        go.Scatter(x=x, y=rolstd ,name = 'rolling_std'))

    fig.update_layout(title_text="Stationary Rolling Window")
    
    return fig

def plot_timerseries_decomposition(df, **kwargs):
    """
    Function that plots the decomposition of the timeseries
    The plot includes the following:
        * Trend values
        * Seasonal values
        * Observed values
        * Residual values
    
    Args:
        df(pd.DataFrame): The timeseries dataframe indexed with date
    Returns:
        fig: Plotly Figure
    """ 
    ts_preprocess.validate_datetime_index(df, allow = False)

    height = kwargs.get('height',800)
    width = kwargs.get('width',1000)
    title_text = kwargs.get('title_text', "Timeseries Components")
    path_html = kwargs.get('path_html', None)



    decomposition = ts_models.calculate_decompose(df, model='multiplicative')

    fig = make_subplots(rows=4, cols=1)
    x = decomposition.observed.index

    fig.add_trace(
        go.Scatter(x=x, y=decomposition.observed.values,name = 'Observed'),
        row=1, col=1)

    fig.add_trace(
        go.Scatter(x=x, y=decomposition.trend.values,name = 'Trend'),
        row=2, col=1)

    fig.add_trace(
        go.Scatter(x=x, y=decomposition.seasonal.values,name = 'Seasonal'),
        row=3, col=1)

    fig.add_trace(
        go.Scatter(x=x, y=decomposition.resid.values,name = 'Residual'),
        row=4, col=1)


    fig.update_layout(height=height, width=width, 
                      title_text=title_text)
   
    if path_html is not None:
        fig.write_html(path_html)
    return fig.show()

def plot_autocorrelation(df, **kwargs):
    """
    Function that plots the autocorrelation of a timeseries
    Args:
        df(pd.DataFrame): The timeseries dataframe indexed with date
    Returns:
        fig: Plotly Figure
    """
    ts_preprocess.validate_datetime_index(df, allow = False)

    nlags = kwargs.get('nlags',40)
    alpha = kwargs.get('alpha',0.05)
    qstat = kwargs.get('qstat',True)
    fft = kwargs.get('fft',False)

    height = kwargs.get('height',600)
    width = kwargs.get('width',1000)
    title_text = kwargs.get('title_text', "Timeseries Components")
    path_html = kwargs.get('path_html', None)


    # Get values for autocorrelation
    cf,confint,qstat,pvalues = acf(df,nlags=nlags,alpha=alpha,qstat = qstat,fft=fft)

    # Get Autocorrelation intervals for plotting
    x = list(range(1,nlags+1))
    y = cf
    y_upper  = 1-confint[:,1]
    y_lower  = (1-confint[:,1])*-1

    # Draw vertical lines
    shapes = list()
    for i in zip(range(1,nlags+1),y):
        shapes.append({'type': 'line',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': i[0],
                    'y0': 0,
                    'x1': i[0],
                    'y1': i[1]})
        
    layout = go.Layout(shapes=shapes)    
    fig = go.Figure([
        go.Scatter(
            x=x,
            y=y,
            fill=None,
            fillcolor='rgba(0,0,255,0)',
            line=dict(color='rgb(0,0,0)'),
            mode='lines+markers',
            name='Autocorrelation'
        ),
        go.Scatter(
            x=x,
            y=y_upper,
            fill='tozeroy',
            fillcolor='rgba(0,0,255,.05)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ),
        go.Scatter(
            x=x,
            y=y_lower,
            fill='tonextx',
            fillcolor='rgba(0,0,255,.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        )
    ])
    fig.update_layout(layout)
    fig.update_layout(title=title_text, xaxis_title = 'Lag', yaxis_title='ACF')
    fig.update_layout(height=height, width=width)

    if path_html is not None:
        fig.write_html(path_html)

    return fig.show()

def plot_timeseries_columns(df):
    """
    Function that plots an interactive Timeseries plot 
    based on the available columns of the dataset
    Args:
        df(pd.DataFrame): The timeseries dataframe indexed with date
    Returns:
        fig: Plotly Figure

    """
    v = ts_preprocess.validate_datetime_index(df, allow = True)

    try:
        column_list = list(df.columns)
    except:
        column_list =[df.name]


    if isinstance(df,pd.Series):
        is_series = True
        xops = ["Series Index"]
        xplaceholder = "NA"
    elif v:
        is_series = True
        xops = ["Series Index"]
        xplaceholder = df.index.name
    else:
        is_series = False
        xops = column_list
        xplaceholder = column_list[0]

    def _plot(df,y,x,title, color):
        """
        Support function that is called when the button is clicked
        """
        if is_series is False:
            fig = px.line(data_frame = df,
                        x=x,
                        y=y,
                        title=title, color = color)
        else:
            fig = px.line(data_frame = df,
                          y=y,
                        title=title)
        return fig.show()
    
    # Create Widgets
    title = widgets.Text(value = " Chart Title",
                         placeholder = "Insert Title Here",
                         description = "Chart title",
                         disabled = False)
    
    timeseries = widgets.Dropdown(options = column_list,
                                  placeholder = column_list[0],
                                  description = "(Y-Axis) Plot Column",
                                  disabled = False)

    xaxis = widgets.Dropdown(options = xops,
                              placeholder = xplaceholder,
                              description = "(X-Axis) Plot Column",
                              disabled = False)
                                

    color_selector = widgets.Dropdown(options = [None] + column_list,
                                     placeholder = None,
                                     description = "Select Color Column",
                                     disabled = False)

    heading = widgets.HBox([title])
    second_row = widgets.HBox([xaxis,timeseries])
    third_row = widgets.HBox([color_selector])
    button = widgets.Button(description = "Generate Chart")
    display(heading)
    display(second_row)
    display(third_row)
    display(button)
    def on_button_clicked(button):
        """
        Function used for button click
        """
        clear_output()
        display(heading)
        display(second_row)
        display(third_row)
        display(button)

        _plot(df,y=timeseries.value,x=xaxis.value,
                 title=title.value, color=color_selector.value)

    button.on_click(functools.partial(on_button_clicked))

def plot_exponential_smoothing_results(train,fcast,test=None,**kwargs):
    """
    Function that plots the results of the function call exponential_smoothing
    """

    height = kwargs.get('height',600)
    width = kwargs.get('width',1000)
    title_text = kwargs.get('title_text', "Model Results Exponential Smoothing")
    path_html = kwargs.get('path_html', None)

    x = train.index

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=train.values,name = 'Train'))

    if test is not None:
        fig.add_trace(
            go.Scatter(x=test.index, y=test.values,name = 'Test'))

    fig.add_trace(
        go.Scatter(x=fcast.index, y=fcast.values,name = 'Forecast'))


    fig.update_layout(height=height, width=width, 
                      title_text=title_text)
   
    if path_html is not None:
        fig.write_html(path_html)

    return fig.show()

def plot_errors(train,fcast,model,**kwargs):

    """
    Support function to plot Residuals, One Step Errors
    """
    height = kwargs.get('height',600)
    width = kwargs.get('width',1000)
    title_text = kwargs.get('title_text', "Residuals & One-Step Error")
    path_html = kwargs.get('path_html', None)

    errors = train.values-shift(fcast.values, 1,cval=train.values[0])
    errors_perc = model.resid/train.values

    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(
        go.Scatter(x=fcast.index, y=model.resid,name = 'Residuals'), row=1,col=1)

    fig.add_trace(
        go.Scatter(x=fcast.index, y=errors,name = 'One Step Error'), row=2,col=1)

    fig.add_trace(
        go.Scatter(x=fcast.index, y=errors_perc,name = 'Resid %'), row=3,col=1)

    fig.update_layout(height=height, width=width, 
                      title_text=title_text)
   
    if path_html is not None:
        fig.write_html(path_html)

    return fig.show()

def plot_scatter_matrix(df, trim_label = None, **kwargs):
    """
    Function that plots the interactive scatter matrix
    """

    height = kwargs.get('height',600)
    width = kwargs.get('width',1000)
    path_html = kwargs.get('path_html', None)

    ops = tuple(df.columns)
    if trim_label is None:
        val = -1
    else:
        val = trim_label

    

    def _plot(df,cols):
        """
        Support function that is called when the button is clicked
        """
        dimensions = []
        for i in cols:
            d = dict(label = i[:val], values = df[i])
            dimensions.append(d)

        fig = go.Figure(data=go.Splom(dimensions=dimensions,
                                      marker=dict(showscale=False,
                                                  line_color='white', line_width=0.5)))

        fig.update_layout(height=height, width=width)
   
        if path_html is not None:
            fig.write_html(path_html)

        return fig.show()
    
    # Create Widgets
    col_widget = widgets.SelectMultiple(options=ops,value=(),rows=5,description='Select Columns:',disabled=False)

    title = widgets.Text(value = " Chart Title",
                         placeholder = "Insert Title Here",
                         description = "Chart title",
                         disabled = False)


    heading = widgets.HBox([title,col_widget])
    button = widgets.Button(description = "Generate Chart")
    display(heading)
    display(button)
    def on_button_clicked(button):
        """
        Function used for button click
        """
        clear_output()
        display(heading)
        display(button)

        _plot(df,cols = col_widget.value)

    button.on_click(functools.partial(on_button_clicked))
