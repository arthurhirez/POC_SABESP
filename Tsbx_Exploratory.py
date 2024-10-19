import numpy as np
import pandas as pd
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose



from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def visualize_full_decomposition(df, variable, start_date, end_date, period = 30, granularity = 'D', model = 'additive',
                            two_sided = False, plot_size = (14, 8), plot_corr = False):
    """
    Visualizes the seasonal decomposition of a specified time series variable within a date range,
    with an option to smooth the series by changing the granularity and choose between additive or multiplicative model.
    Includes an autocorrelation plot and combined trend + seasonal + autocorrelation analysis.

    Parameters:
    df (pd.DataFrame): DataFrame with a DatetimeIndex and numerical variables.
    variable (str): The column name of the variable to decompose.
    start_date (str): The start date of the period (format: 'YYYY-MM-DD').
    end_date (str): The end date of the period (format: 'YYYY-MM-DD').
    period (int): The period to use for decomposition (default is 30).
    granularity (str): The resampling frequency (default is 'D' for daily).
    model (str): The model for decomposition ('additive' or 'multiplicative'; default is 'additive').

    Returns:
    None: Displays the decomposition plots and autocorrelation analysis.
    """

    # Ensure the variable exists in the DataFrame
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in DataFrame.")
        return

    # Filter the DataFrame for the specified date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask]

    # Check if the filtered DataFrame is not empty
    if df_filtered.empty:
        print(f"No data found for the specified date range: {start_date} to {end_date}.")
        return

    # Resample the data based on the specified granularity and take the mean
    df_smoothed = df_filtered[variable].resample(granularity).mean()

    # Decomposition
    decomposition = seasonal_decompose(df_smoothed, model = model, period = period, two_sided = two_sided)

    # Set plot size
    plt.rcParams.update({'figure.figsize': plot_size})

    if not plot_corr:
        # Plot the chosen decomposition (Original, Trend, Seasonal, Residual)
        fig_decomposition = decomposition.plot()
        fig_decomposition.suptitle(f'{model.capitalize()} Decomposition', fontsize = 16)
        plt.tight_layout()  # Adjust layout


    # Draw Plot
    fig, axes = plt.subplots(1, 2, figsize = (16, 3), dpi = 100)
    plot_acf(df_smoothed.dropna(), lags = 50, alpha = 0.05,  ax = axes[0])
    plot_pacf(df_smoothed.dropna(), lags = 50, alpha = 0.05,  ax = axes[1])

    plt.tight_layout()
    plt.show()

    if not plot_corr:
        # Calculate and plot the sum of trend + seasonal
        trend_plus_seasonal = decomposition.trend + decomposition.seasonal

        plt.figure(figsize = plot_size)
        plt.plot(df_smoothed.index, df_smoothed, label = 'Original Series', color = 'blue', alpha = 0.5)
        plt.plot(trend_plus_seasonal.index, trend_plus_seasonal, label = 'Trend + Seasonal', color = 'orange')
        plt.title('Trend + Seasonal Components', fontsize = 16)
        plt.xlabel('Date', fontsize = 14)
        plt.ylabel(variable, fontsize = 14)
        plt.legend()
        plt.tight_layout()
        plt.show()






def visualize_timeseries(df, start_date, end_date, variable, granularity = 'H'):
    """
    Visualizes the time series data for a specific variable and time period, with adjustable granularity.

    Parameters:
    df (pd.DataFrame): DataFrame with a DatetimeIndex and numerical variables.
    start_date (str): The start date of the period (format: 'YYYY-MM-DD').
    end_date (str): The end date of the period (format: 'YYYY-MM-DD').
    variable (str): The column name of the variable to plot.
    granularity (str): The resampling frequency (e.g., 'H' for hourly, 'D' for daily, 'W' for weekly, etc.).

    Returns:
    None: Displays the plot.
    """

    # Filter the DataFrame for the specified time period
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask]

    # Check if the variable exists in the DataFrame
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in DataFrame.")
        return

    # Resample the data based on the chosen granularity
    df_resampled = df_filtered[variable].resample(granularity).mean()

    # Plot the time series data
    plt.figure(figsize = (10, 6))
    plt.plot(df_resampled.index, df_resampled, marker = 'o', linestyle = '-')

    # Set plot title and labels
    plt.title(f'Time Series of {variable} from {start_date} to {end_date} ({granularity} granularity)', fontsize = 14)
    plt.xlabel('Date', fontsize = 12)
    plt.ylabel(variable, fontsize = 12)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation = 45)

    # Show grid and plot
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()

def visualize_decomposition(df, variable, start_date, end_date, period = 30, granularity = 'D', model = 'additive',
                            two_sided = False, plot_size = (14, 8)):
    """
    Visualizes the seasonal decomposition of a specified time series variable within a date range,
    with an option to smooth the series by changing the granularity and choose between additive or multiplicative model.

    Parameters:
    df (pd.DataFrame): DataFrame with a DatetimeIndex and numerical variables.
    variable (str): The column name of the variable to decompose.
    start_date (str): The start date of the period (format: 'YYYY-MM-DD').
    end_date (str): The end date of the period (format: 'YYYY-MM-DD').
    period (int): The period to use for decomposition (default is 30).
    granularity (str): The resampling frequency (default is 'D' for daily).
        Common options include:
            'H': Hourly
            'D': Daily (default)
            'W': Weekly
            'M': Monthly
    model (str): The model for decomposition ('additive' or 'multiplicative'; default is 'additive').

    Returns:
    None: Displays the decomposition plots.
    """

    # Ensure the variable exists in the DataFrame
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in DataFrame.")
        return

    # Filter the DataFrame for the specified date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask]

    # Check if the filtered DataFrame is not empty
    if df_filtered.empty:
        print(f"No data found for the specified date range: {start_date} to {end_date}.")
        return

    # Resample the data based on the specified granularity and take the mean
    df_smoothed = df_filtered[variable].resample(granularity).mean()

    # Decomposition
    decomposition = seasonal_decompose(df_smoothed, model = model, period = period, two_sided = two_sided)

    # Set plot size
    plt.rcParams.update({'figure.figsize': plot_size})

    # Plot the chosen decomposition
    fig_decomposition = decomposition.plot()
    fig_decomposition.suptitle(f'{model.capitalize()} Decomposition', fontsize = 16)
    plt.tight_layout()  # Adjust layout

    # Calculate and plot the sum of trend + seasonal
    trend_plus_seasonal = decomposition.trend + decomposition.seasonal

    plt.figure(figsize = plot_size)
    plt.plot(df_smoothed.index, df_smoothed, label = 'Original Series', color = 'blue', alpha = 0.5)
    plt.plot(trend_plus_seasonal.index, trend_plus_seasonal, label = 'Trend + Seasonal', color = 'orange')
    plt.title('Trend + Seasonal Components', fontsize = 16)
    plt.xlabel('Date', fontsize = 14)
    plt.ylabel(variable, fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.show()


def transform_time_series(data, transformation = 'none'):
    """
    Apply a specified transformation to a time series data.

    Parameters:
        data (pd.Series): The time series data to transform.
        transformation (str): The type of transformation to apply. Options are
                              'box-cox', 'log', 'sqrt', and 'none'.

    Returns:
        pd.Series: The transformed time series.
    """
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")

    if transformation == 'box-cox':
        # Box-Cox transformation requires all values to be positive
        if (data <= 0).any():
            raise ValueError("All values must be positive for Box-Cox transformation.")
        transformed_data, lambda_ = stats.boxcox(data)
        print('box')
        return pd.Series(transformed_data, index = data.index), lambda_
        print('cox')
    elif transformation == 'log':
        if (data <= 0).any():
            raise ValueError("All values must be positive for log transformation.")
        transformed_data = np.log(data)
        return pd.Series(transformed_data, index = data.index)

    elif transformation == 'sqrt':
        transformed_data = np.sqrt(data)
        return pd.Series(transformed_data, index = data.index)

    elif transformation == 'none':
        return data

    else:
        raise ValueError("Invalid transformation specified. Choose from 'box-cox', 'log', 'sqrt', or 'none'.")



import pandas as pd
import numpy as np
from scipy import stats

def transform_time_series_diff(data, transformation='none', granularity = 'M', diff_order=0):
    """
    Apply a specified transformation to a time series data, including differentiation.

    Parameters:
        data (pd.Series): The time series data to transform.
        transformation (str): The type of transformation to apply. Options are
                              'box-cox', 'log', 'sqrt', 'none'.
        diff_order (int): The order of differencing to apply (0 for no differencing).

    Returns:
        pd.Series: The transformed time series.
    """
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")

    # Apply specified transformation
    if transformation == 'box-cox':
        # Box-Cox transformation requires all values to be positive
        if (data <= 0).any():
            raise ValueError("All values must be positive for Box-Cox transformation.")
        transformed_data, lambda_ = stats.boxcox(data)
        data = pd.Series(transformed_data, index=data.index)
    elif transformation == 'log':
        if (data <= 0).any():
            raise ValueError("All values must be positive for log transformation.")
        data = np.log(data)
    elif transformation == 'sqrt':
        data = np.sqrt(data)
    elif transformation != 'none':
        raise ValueError("Invalid transformation specified. Choose from 'box-cox', 'log', 'sqrt', or 'none'.")

    # Apply differencing
    if diff_order > 0:
        df_smoothed = data.resample(granularity).mean()
        data = df_smoothed.diff(periods=diff_order).dropna()

    return data

# Example usage
# transformed_data = transform_time_series(your_series, transformation='log', diff_order=1)


def plot_hued_by_year(df, variable, start_date, end_date, granularity = 'D'):
    """
    Plots a time series line plot hued by year, for a given time period and granularity.
    The x-axis shows the months, and each year is plotted on the same time scale.

    Parameters:
    df (pd.DataFrame): DataFrame with a DatetimeIndex and numerical variables.
    variable (str): The column name of the variable to plot.
    start_date (str): The start date of the period (format: 'YYYY-MM-DD').
    end_date (str): The end date of the period (format: 'YYYY-MM-DD').
    granularity (str): The resampling frequency (default is 'D' for daily).
        Common options include:
            'H': Hourly
            'D': Daily (default)
            'W': Weekly
            'M': Monthly

    Returns:
    None: Displays the line plot hued by year.
    """

    # Ensure the variable exists in the DataFrame
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in DataFrame.")
        return

    # Filter the DataFrame for the specified date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask]

    # Check if the filtered DataFrame is not empty
    if df_filtered.empty:
        print(f"No data found for the specified date range: {start_date} to {end_date}.")
        return

    df_filtered.index = pd.to_datetime(df_filtered.index)

    # Resample the data based on the specified granularity
    df_resampled = df_filtered.resample(granularity).mean()

    # Extract the year, month, and day for plotting
    df_resampled['Year'] = df_resampled.index.year
    df_resampled['Month-Day'] = df_resampled.index.strftime('%m-%d')  # Ignore the year part, keep month-day

    # Plot using Seaborn
    plt.figure(figsize = (16, 8))
    sns.lineplot(data = df_resampled, x = 'Month-Day', y = variable, hue = 'Year', palette = 'magma')

    plt.title(f'{variable} Trend by Year (Aligned by Month-Day)', fontsize = 16)
    plt.xlabel('Month-Day', fontsize = 14)
    plt.ylabel(variable, fontsize = 14)
    plt.xticks(rotation = 45)  # Rotate x-axis labels for better readability
    plt.legend(title = 'Year')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_monthly_facets(df, variable, start_date, end_date, granularity = 'D', specific_years = [2014, 2018, 2022]):
    """
    Plots 12 subplots, one for each month, showing the trend of the specified variable for each year.

    Parameters:
    df (pd.DataFrame): DataFrame with a DatetimeIndex and numerical variables.
    variable (str): The column name of the variable to plot.
    start_date (str): The start date of the period (format: 'YYYY-MM-DD').
    end_date (str): The end date of the period (format: 'YYYY-MM-DD').
    granularity (str): The resampling frequency (default is 'D' for daily).
    specific_years (list): A list of specific years to show as x-axis ticks (e.g., [2014, 2018, 2022]).

    Returns:
    None: Displays the line plots.
    """

    # Ensure the variable exists in the DataFrame
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in DataFrame.")
        return

    # Filter the DataFrame for the specified date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask]

    # Check if the filtered DataFrame is not empty
    if df_filtered.empty:
        print(f"No data found for the specified date range: {start_date} to {end_date}.")
        return

    # Resample the data based on the specified granularity
    df_resampled = df_filtered.resample(granularity).mean()

    # Extract the year and month for plotting
    df_resampled['Year'] = df_resampled.index.year
    df_resampled['Month'] = df_resampled.index.month

    # Create a FacetGrid for 12 months
    g = sns.FacetGrid(df_resampled, col = 'Month', col_wrap = 12, height = 4, aspect = .5)

    # Map the lineplot to each facet
    g = g.map(sns.lineplot, 'Year', variable, marker = "o")

    # Control the x-axis ticker labels
    if specific_years is not None:
        for ax in g.axes.flatten():
            ax.set_xticks(specific_years)  # Set specific ticks
            ax.set_xticklabels(specific_years, rotation = 45)  # Set the labels and rotate them

    # Set titles and axis labels
    g.set_titles(col_template = "{col_name}")
    g.set_axis_labels('Year', variable)

    # Adjust layout
    plt.tight_layout()
    plt.show()
