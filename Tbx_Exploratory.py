import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


def create_labels(bins):
    labels = []
    for i in range(len(bins) - 1):
        if bins[i + 1] == np.inf:
            labels.append(f'{bins[i]}+')
        else:
            labels.append(f'{bins[i]}-{bins[i + 1]}')
    return labels


def plot_time_series_with_resampling(data, tgt_val, granularity, agg_func = 'mean', date_range = None, bins = None):
    df = data.sort_values(by = 'data_lancamento').copy()

    df['data_lancamento'] = pd.to_datetime(df['data_lancamento'])

    # Set the 'data_lancamento' column as the index
    df.set_index('data_lancamento', inplace = True)

    if date_range:
        start_date, end_date = date_range
        df = df[start_date:end_date]

    # Dictionary to map string function names to actual pandas functions
    agg_funcs = {
        'mean': 'mean',
        'count': 'count',
        'sum': 'sum',
        'min': 'min',
        'max': 'max'
        }

    # Check if the provided agg_func is valid
    if agg_func not in agg_funcs:
        raise ValueError(f"Invalid agg_func. Choose from {list(agg_funcs.keys())}")

    # Resample the data using the specified aggregation function
    resampled_data = df[tgt_val].resample(granularity).agg(agg_funcs[agg_func]).reset_index()

    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize = (12, 6))

    # Create bins for the tgt_val column
    if bins is None:
        bins = [0, 50, 100, 200, 300, np.inf]

    labels = create_labels(bins)
    df['m2_tipo'] = pd.cut(df[tgt_val], bins = bins, labels = labels)
    aux_df = df[['m2_tipo'] + [tgt_val]].sort_values(by = 'm2_tipo')

    for bin in aux_df['m2_tipo'].unique():
        aux_bins = aux_df[aux_df['m2_tipo'] == bin]
        aux_monthly_mean = aux_bins[tgt_val].resample(granularity).agg(agg_funcs[agg_func]).reset_index()
        sns.lineplot(data = aux_monthly_mean, x = 'data_lancamento', y = tgt_val, label = bin, ax = ax)

    # Plot the overall resampled data
    sns.lineplot(data = resampled_data, x = 'data_lancamento', y = tgt_val,
                 label = f'{agg_func.capitalize()} of {tgt_val}', color = 'black',
                 lw = 2.5, ls = 'dashed', ax = ax)

    # Customize the plot
    plt.title('Time Series Data and Resampled Data')
    plt.xlabel('Date')
    plt.ylabel('Quantidade lançamentos')
    plt.legend()
    plt.show()


def boxplots_idade(data_full, data_nan, bins = None):
    if bins is None:
        bins = [0, 50, 100, 200, 300, np.inf]

    labels = create_labels(bins)

    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize = (14, 6))

    # First boxplot
    sns.boxplot(data = data_full, x = 'Idade_predio',
                hue = pd.cut(data_full['M2_total_unidade_tipo'],
                             bins = bins,
                             labels = labels),
                gap = 0.2, ax = axes[0])  # Specify the second subplot axis

    # Set title for the first plot
    axes[0].set_title('Idade de todos os prédios da base')
    axes[0].get_legend().remove()
    # Second boxplot
    sns.boxplot(data = data_nan, x = 'Idade_predio',
                hue = pd.cut(data_nan['M2_total_unidade_tipo'],
                             bins = bins,
                             labels = labels),
                gap = 0.2, ax = axes[1])  # Specify the first subplot axis

    # Set only observed values on x-axis
    observed_values = data_nan['Idade_predio'].dropna().unique()
    axes[1].set_xticks(observed_values)

    # Move the legend outside the figure for the second plot
    axes[1].legend(title = 'M2_total_unidade_tipo', bbox_to_anchor = (1.05, 1), loc = 'upper left')

    # Set title for the second plot
    axes[1].set_title('Idade dos prédios com valores faltantes de metragem')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def merge_with_tolerance(df1, df2, id_dfs, agg_cols, var_col, tolerance):
    # First, do an exact merge on the 'CEP' column
    df_merged = pd.merge(df1, df2, on = agg_cols, how = 'inner', suffixes = (id_dfs[0], id_dfs[1]))

    # Filter rows where the 'NUMERO_RUA' column from df1 is within the tolerance range of df2
    df_merged = df_merged[(df_merged[f'{var_col}' + id_dfs[0]] >= df_merged[f'{var_col}' + id_dfs[1]] - tolerance) &
                          (df_merged[f'{var_col}' + id_dfs[0]] <= df_merged[f'{var_col}' + id_dfs[1]] + tolerance)]

    # Reordering columns
    agg_col_suffixes = [f'{col}{id_dfs[0]}' for col in agg_cols] + [f'{col}{id_dfs[1]}' for col in agg_cols if
                                                                    f'{col}{id_dfs[1]}' in df_merged.columns]
    var_col_suffixes = [f'{var_col}{id_dfs[0]}', f'{var_col}{id_dfs[1]}']

    # Get the rest of the columns
    other_cols = [col for col in df_merged.columns if col not in agg_col_suffixes + var_col_suffixes]
    other_cols = [c for c in other_cols if c not in agg_cols]
    # New order: agg_cols + var_col_suffixes + other columns
    new_order = agg_cols + var_col_suffixes + other_cols

    # Reorder columns
    df_merged = df_merged[new_order]

    return df_merged


def merge_df_tolerance(data_cep_iptu, data_iptu, data_imoveis, tol_NUMERO = 10, tol_PVTOS = 5, tol_OBSOLENCIA = 0.1):
    qwe = merge_with_tolerance(df1 = data_cep_iptu,
                               df2 = data_iptu[['DESC_RUA', 'NUMERO_RUA', 'CEP', 'QUANTIDADE DE PAVIMENTOS',
                                                'FATOR DE OBSOLESCENCIA', 'TIPO DE PADRAO DA CONSTRUCAO',
                                                'TESTADA PARA CALCULO', 'VALOR DO M2 DO TERRENO']],
                               id_dfs = ['_cep', '_iptu'],
                               agg_cols = ['CEP'],
                               var_col = 'NUMERO_RUA',
                               tolerance = tol_NUMERO)

    qwe['QUANTIDADE DE PAVIMENTOS'] = qwe['QUANTIDADE DE PAVIMENTOS'].astype(int)

    qwe['Imovel_residencial'] = np.where(qwe['TIPO DE PADRAO DA CONSTRUCAO'].str.contains('Residencial'), 1, 0)

    # Create 'Imovel_vertical' column: 1 if 'vertical' is in the string, 0 otherwise
    qwe['Imovel_vertical'] = np.where(qwe['TIPO DE PADRAO DA CONSTRUCAO'].str.contains('vertical'), 1, 0)

    # Create a new column for the remaining part ('padrão X')
    qwe['Padrao'] = qwe['TIPO DE PADRAO DA CONSTRUCAO'].str.extract(r'(padrão \w)')
    qwe['Padrao'] = qwe['Padrao'].str.replace('padrão ', '')

    asd = merge_with_tolerance(df1 = qwe.rename(columns = {'DESC_RUA_cep': 'DESC_RUA',
                                                           'NUMERO_RUA_cep': 'NUMERO_RUA',
                                                           'QUANTIDADE DE PAVIMENTOS': 'Andares_tipo'}),
                               df2 = data_imoveis[
                                   ['DESC_RUA', 'NUMERO_RUA', 'Imovel_residencial', 'Imovel_vertical', 'Andares_tipo',
                                    'Total_Unidades', 'Idade_predio', 'Blocos',
                                    'M2_util_unidade_tipo', 'M2_total_unidade_tipo', 'RS_por_M2_area_util_IGPM',
                                    'RS_por_M2_area_total_IGPM', ]],
                               id_dfs = ['_cep_iptu', '_cg'],
                               agg_cols = ['DESC_RUA', 'NUMERO_RUA', 'Imovel_residencial', 'Imovel_vertical'],
                               var_col = 'Andares_tipo',
                               tolerance = tol_PVTOS).drop_duplicates()

    asd = asd[asd['Imovel_vertical'] == 1]  # muitas incertezas com horizontais

    # DataFrame containing obsolescence factors
    df_factors = pd.DataFrame({
        'Idade do Prédio (em anos)': ["Menor que 1"] + list(range(1, 43)),
        'Fatores de Obsolescência para padrões A e B': [1.00, 0.99, 0.98, 0.97, 0.96, 0.94, 0.93, 0.92, 0.90, 0.89,
                                                        0.88, 0.86, 0.84, 0.83, 0.81, 0.79, 0.78, 0.76, 0.74, 0.72,
                                                        0.70, 0.68, 0.66, 0.64, 0.62, 0.59, 0.57, 0.55, 0.52, 0.50,
                                                        0.48, 0.45, 0.42, 0.40, 0.37, 0.34, 0.32, 0.29, 0.26, 0.23,
                                                        0.20, 0.20, 0.20],
        'Fatores de Obsolescência para demais padrões e tipos': [1.00, 0.99, 0.99, 0.98, 0.97, 0.96, 0.96, 0.95, 0.94,
                                                                 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.88, 0.86, 0.85,
                                                                 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.76, 0.75,
                                                                 0.74, 0.73, 0.71, 0.70, 0.69, 0.67, 0.66, 0.64, 0.63,
                                                                 0.62, 0.60, 0.59, 0.57, 0.56, 0.54, 0.52]
        })

    def get_obsolescence_factor(age, pattern, df_factors):
        # Map age to lookup value
        if age < 1:
            age_lookup = "Menor que 1"
        else:
            age_lookup = age

        # Determine the column to use based on the pattern
        if pattern in ['A', 'B']:
            factor_column = 'Fatores de Obsolescência para padrões A e B'
        else:
            factor_column = 'Fatores de Obsolescência para demais padrões e tipos'

        # Perform the lookup and handle missing results
        factors = df_factors[df_factors['Idade do Prédio (em anos)'] == age_lookup]

        if not factors.empty:
            factor = factors[factor_column].values[0]
        else:
            # Handle the case where no matching age is found (you can adjust this as needed)
            factor = -1  # Or set a default factor value

        return factor

    # Apply the function to your DataFrame
    asd['Fator_obsolescencia_calculado'] = asd.apply(
        lambda row: get_obsolescence_factor(row['Idade_predio'], row['Padrao'], df_factors), axis = 1)

    asd = asd[abs(asd['Fator_obsolescencia_calculado'] - asd['FATOR DE OBSOLESCENCIA'].astype(float)) < tol_OBSOLENCIA]

    asd['TOL_NUM'] = abs(asd['NUMERO_RUA'] - asd['NUMERO_RUA_iptu'])

    asd['TESTADA PARA CALCULO'] = asd['TESTADA PARA CALCULO'].astype(float)

    return asd[['DESC_RUA', 'NUMERO_RUA', 'NUMERO_RUA_iptu', 'Andares_tipo_cep_iptu', 'Andares_tipo_cg', 'TOL_NUM',
                'TESTADA PARA CALCULO',
                'Fator_obsolescencia_calculado', 'FATOR DE OBSOLESCENCIA', 'Total_Unidades', 'Blocos',
                'M2_util_unidade_tipo', 'M2_total_unidade_tipo', 'RS_por_M2_area_util_IGPM',
                'RS_por_M2_area_total_IGPM', 'VALOR DO M2 DO TERRENO', 'TIPO DE PADRAO DA CONSTRUCAO']].sort_values(by = ['DESC_RUA', 'NUMERO_RUA'])




def scrape_ceps_from_queries(query_list):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Set up the WebDriver (adjust the path to where your ChromeDriver is located)
    service = Service('C:\\Users\\arthu\\USPy\\chromeDriver\\chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=chrome_options)

    results = []
    count_err = 1
    count = 0

    # Iterate through the list of queries
    for idx, query in enumerate(query_list):
        try:
            print(f"Processing {idx+1}/{len(query_list)}: {query}")

            # Search for the query in Google Maps
            search_url = f"https://www.google.com/maps/search/{query}"
            driver.get(search_url)

            # Wait for the results to load
            time.sleep(3)  # Adjust wait time if necessary

            # Locate the address information (span tag with class 'DkEaL')
            address_info = driver.find_element(By.CLASS_NAME, 'Io6YTe').text

            # Append the result (query, address_info)
            results.append((query, address_info))

            # Print the extracted information
            print(f"Extracted: {address_info}")

        except Exception as e:
            print(f"{count_err} / {count}- Error processing {query}")
            results.append((query, "Error"))
            count_err += 1

        count += 1

    # Close the browser after scraping
    driver.quit()

    return results


def get_obsolescence(age, pattern):
    # DataFrame containing obsolescence factors
    df_factors = pd.DataFrame({
        'Idade do Prédio (em anos)': ["Menor que 1"] + list(range(1, 43)),
        'Fatores de Obsolescência para padrões A e B': [1.00, 0.99, 0.98, 0.97, 0.96, 0.94, 0.93, 0.92, 0.90, 0.89,
                                                        0.88, 0.86, 0.84, 0.83, 0.81, 0.79, 0.78, 0.76, 0.74, 0.72,
                                                        0.70, 0.68, 0.66, 0.64, 0.62, 0.59, 0.57, 0.55, 0.52, 0.50,
                                                        0.48, 0.45, 0.42, 0.40, 0.37, 0.34, 0.32, 0.29, 0.26, 0.23,
                                                        0.20, 0.20, 0.20],
        'Fatores de Obsolescência para demais padrões e tipos': [1.00, 0.99, 0.99, 0.98, 0.97, 0.96, 0.96, 0.95, 0.94,
                                                                 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.88, 0.86, 0.85,
                                                                 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.76, 0.75,
                                                                 0.74, 0.73, 0.71, 0.70, 0.69, 0.67, 0.66, 0.64, 0.63,
                                                                 0.62, 0.60, 0.59, 0.57, 0.56, 0.54, 0.52]
        })
    # Map age to lookup value
    if age < 1:
        age_lookup = "Menor que 1"
    else:
        age_lookup = age

    # Determine the column to use based on the pattern
    if pattern in ['A', 'B']:
        factor_column = 'Fatores de Obsolescência para padrões A e B'
    else:
        factor_column = 'Fatores de Obsolescência para demais padrões e tipos'

    # Perform the lookup and handle missing results
    factors = df_factors[df_factors['Idade do Prédio (em anos)'] == age_lookup]

    if not factors.empty:
        factor = factors[factor_column].values[0]
    else:
        # Handle the case where no matching age is found (you can adjust this as needed)
        factor = -1  # Or set a default factor value

    return factor


def convert_padrao_iptu(data, drop_original_col = True, compute_obsolence = False):
    # Create a new column for the remaining part ('padrão X')
    data['Padrao'] = data['TIPO DE PADRAO DA CONSTRUCAO'].str.extract(r'(padrão \w)')
    data['Padrao'] = data['Padrao'].str.replace('padrão ', '')

    if drop_original_col:
        data.drop(columns = 'TIPO DE PADRAO DA CONSTRUCAO', inplace = True)
    if compute_obsolence:
        # Apply the function to your DataFrame
        data['Fator_obsolescencia_calculado'] = data.apply(
            lambda row: get_obsolescence(row['Idade_predio'], row['Padrao']), axis = 1)
    return data