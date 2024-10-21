import numpy as np
import pandas as pd
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def filter_resample_TS(df, start_date, end_date, granularity = 'D', hue_by = None, agg_func = 'mean'):
    # Filter the DataFrame for the specified date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask]

    # Check if the filtered DataFrame is not empty
    if df_filtered.empty:
        print(f"No data found for the specified date range: {start_date} to {end_date}.")
        return

    # Define a dictionary of possible aggregation functions
    agg_functions = {
        'mean': 'mean',
        'sum': 'sum',
        'min': 'min',
        'max': 'max',
        'median': 'median',
        'std': 'std'
        }

    # Ensure the provided agg_func is valid
    if agg_func not in agg_functions:
        print(f"Invalid aggregation function: {agg_func}. Using default 'mean'.")
        agg_func = 'mean'

    # Resample the data based on the specified granularity and aggregation function
    if hue_by is None:
        df_filtered = df_filtered.select_dtypes(include = 'number')
        df_resampled = df_filtered.resample(granularity).agg(agg_functions[agg_func])
    else:
        df_resampled = df_filtered.groupby(hue_by).resample(granularity).agg(agg_functions[agg_func])

    df_resampled = df_resampled.reset_index()
    df_resampled['timestamp'] = df_resampled['timestamp'].replace('2024-02-29', '2024-02-28')

    # Formatting date for certain granularities
    if 'M' in granularity:
        df_resampled['DATE'] = df_resampled['timestamp'].dt.strftime('%m-%d')  # Month-Day format
    elif 'W' in granularity:
        df_resampled['DATE'] = df_resampled['timestamp'].dt.strftime('%m-%U')  # Month-Week number format

    df_resampled['YEAR'] = df_resampled['timestamp'].dt.year.astype(str)
    df_resampled = df_resampled.sort_values('timestamp')

    return df_resampled


def smooth_consumption(data,
                       tgt_feat = 'QTD_CONSUMO',
                       ref_feat = 'QTD_CONSUMO_MEDIO',
                       threshold = 3, limit_status = 2.75):
    # required_columns = ['PDE', 'QTD_CONSUMO', 'QTD_CONSUMO_MEDIO']
    # if all(col in data.columns for col in required_columns):
    #     data_clean = data[required_columns].drop_duplicates()
    # else:
    #     data_clean = data
    # First: Create a column with values 'Atenção' or 'Normal'
    data_tgt = data.copy()

    data_tgt['Status'] = data_tgt.apply(
        lambda row: 'Atenção' if (row[tgt_feat] > limit_status * row[ref_feat] and row[ref_feat] >= threshold)
        else 'Normal', axis = 1
        )

    # Second: Create a new column, copying QTD_CONSUMO but replacing values where QTD_CONSUMO > 10ref_feat
    data_tgt['QTD_CONSUMO_Adjusted'] = data_tgt.apply(
        lambda row: row[ref_feat] if (row[tgt_feat] > limit_status * row[ref_feat] and row[ref_feat] >= threshold)
        else row[tgt_feat], axis = 1)

    print(f"Casos ATENÇÃO!!\t{len(data_tgt[data_tgt['Status'] != 'Normal'])}")
    return data_tgt

def exploratory_processing(data, pde_list, remove_zero_consumption = False, factor_remove = 10, plot_result = False,
                           return_removed = False):

    data_tgt = data[data['COD_PDE'].isin(pde_list)].copy()

    print(f'Quantidade de consumidores: {len(data_tgt["PDE"].unique())}')
    print(f'Quantidade de observações: {data_tgt.shape[0]}')

    median_consumption = data_tgt.groupby('PDE')['QTD_CONSUMO'].median().to_frame('Consumo_Mediano')

    data_tgt = data_tgt.reset_index()
    data_tgt = data_tgt.merge(median_consumption, on = 'PDE', how = 'left')
    data_tgt.set_index('timestamp', inplace = True)
    data_tgt.sort_index(inplace = True)

    data_tgt = data_tgt[['PDE', 'QTD_CONSUMO', 'QTD_CONSUMO_MEDIO', 'Consumo_Mediano']]

    data_agg = data_tgt.groupby('PDE').mean()
    remove_pde = filter_outliers_dynamic_mean(data_agg, 'QTD_CONSUMO', factor = factor_remove)

    data_clean = data_tgt[~data_tgt['PDE'].isin(remove_pde)]

    if remove_zero_consumption:
        # data_clean['PDE'] = data_clean['PDE'].astype(int)
        # data_clean = data_clean.groupby('PDE').resample(granularity).mean().reset_index()
        data_clean = data_clean[(data_clean['QTD_CONSUMO'] != 0) & (data_clean['QTD_CONSUMO_MEDIO'] != 0)]

    if plot_result:
        plt.figure(figsize = (15, 4))  # Control the size of the plot here

        # Create the histogram
        ax = sns.histplot(data = data_clean, x = data_clean.index, y = 'QTD_CONSUMO', bins = 100, stat = 'count',
                          alpha = 0.5)

        # Calculate the mean QTD_CONSUMO for each date (x value) & Plot the mean line
        mean_values = data_clean.groupby(data_clean.index)['QTD_CONSUMO'].mean()
        plt.plot(mean_values.index, mean_values.values, color = 'red', linestyle = '--', label = 'Mean QTD_CONSUMO')

        # Calculate the median QTD_CONSUMO for each date (x value) & Plot the median line
        median_values = data_clean.groupby(data_clean.index)['QTD_CONSUMO'].median()
        plt.plot(median_values.index, median_values.values, color = 'blue', linestyle = '--',
                 label = 'Median QTD_CONSUMO')

        # Add labels and title if needed
        plt.xlabel('Date')
        plt.ylabel('QTD_CONSUMO')
        plt.title('Histogram of QTD_CONSUMO with Mean Line')
        plt.legend()
        # Show the plot
        plt.show()

    if return_removed: return data_clean, remove_pde
    return data_clean


def filter_outliers_dynamic_mean(df, column, factor = 50):
    filtered_df = df.copy()
    excluded_cases = pd.DataFrame()  # DataFrame to hold excluded cases

    while True:
        # Step 1: Calculate the mean of the remaining values
        mean_consumption = filtered_df[column].mean()

        # Step 2: Define the outlier threshold
        outlier_threshold = mean_consumption * factor

        # Step 3: Identify outliers
        outliers = filtered_df[filtered_df[column] > outlier_threshold]

        # Step 4: Filter out outliers
        new_filtered_df = filtered_df[filtered_df[column] <= outlier_threshold]

        # Step 5: Append the outliers to the excluded_cases DataFrame
        excluded_cases = pd.concat([excluded_cases, outliers])

        # If no change in DataFrame, break the loop
        if new_filtered_df.shape[0] == filtered_df.shape[0]:
            break

        filtered_df = new_filtered_df  # Update filtered DataFrame

    print(excluded_cases.reset_index()['PDE'].tolist())

    return excluded_cases.reset_index()['PDE'].tolist()


def plot_hued_by_year(data, variable, start_date, end_date, granularity = 'D', adjust_consumption = True,
                      exclude_case = None):
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
    if variable not in data.columns:
        print(f"Variable '{variable}' not found in DataFrame.")
        return

    df_resampled = filter_resample_TS(data, start_date, end_date, granularity)

    if isinstance(exclude_case, dict):
        for var, value in exclude_case.items():
            # print(var, value)
            df_resampled = df_resampled[df_resampled[var] != value]

    # Extract the year, month, and day for plotting
    # df_resampled['Year'] = df_resampled.index.year
    # df_resampled['Month-Day'] = df_resampled.index.strftime('%m-%d').astype(str)  # Ignore the year part, keep month-day

    # Plot using Seaborn

    if adjust_consumption:
        fig, axs = plt.subplots(2, 1, figsize = (12, 6), sharey = True)  # sharey=True for shared y-axis

        # First plot with title
        sns.lineplot(data = df_resampled, x = 'DATE', y = 'QTD_CONSUMO', hue = 'YEAR', palette = 'magma', ax = axs[0])
        axs[0].set_title('Consumption over Time')

        # Second plot with title
        sns.lineplot(data = df_resampled, x = 'DATE', y = 'QTD_CONSUMO_Adjusted', hue = 'YEAR', palette = 'magma',
                     ax = axs[1])
        axs[1].set_title('Adjusted Consumption over Time')

    else:
        plt.figure(figsize = (16, 8))
        sns.lineplot(data = df_resampled, x = 'DATE', y = variable, hue = 'YEAR', palette = 'magma')
    # Set x-axis limits to start and end at the beginning and end of the year
    # plt.xlim('01-01', '12-31')  # Limit x-axis to month-day format

    plt.title(f'{variable} Trend by Year (Aligned by Month-Day)', fontsize = 16)
    plt.xlabel('Month-Day', fontsize = 14)
    plt.ylabel(variable, fontsize = 14)
    plt.xticks(rotation = 45)  # Rotate x-axis labels for better readability
    plt.legend(title = 'Year')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
