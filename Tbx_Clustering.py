import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from Tsbx_Exploratory import filter_resample_TS


def scale_data(X, y = None, normalize = 'standard', split = False, test_size = 0.2, stratify = None, random_state = 42):
    # Apply normalization or standardization
    if normalize == 'standard':
        scaler = StandardScaler()
    elif normalize == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid normalization type. Choose 'standard' or 'minmax'.")

    # If splitting is needed
    if split:
        if y is not None:
            # Split both X and y
            if stratify is None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size,
                                                                    random_state = random_state)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = stratify,
                                                                    random_state = random_state)

            # Scale the features only (not y)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Convert scaled data back to DataFrame with original column names and indexes
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = X.columns, index = X_train.index)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X.columns, index = X_test.index)

            return X_train_scaled_df, X_test_scaled_df, y_train, y_test
        else:
            # Only split X if y is not provided
            if stratify is None:
                X_train, X_test = train_test_split(X, test_size = test_size, random_state = random_state)
            else:
                X_train, X_test = train_test_split(X, test_size = test_size, stratify = stratify,
                                                   random_state = random_state)

            # Scale the features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Convert scaled data back to DataFrame with original column names and indexes
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = X.columns, index = X_train.index)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X.columns, index = X_test.index)

            return X_train_scaled_df, X_test_scaled_df
    else:
        # No splitting, just scaling
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns, index = X.index)  # Keep original index

        return X_scaled_df


def prototype_Kmeans(data, num_clusters, num_prototypes = 1):
    """
    Clusters consumers based on their consumption series and returns representative examples.

    Parameters:
    - data (pd.DataFrame): DataFrame containing consumption series with 'PDE' as a categorical column.
    - num_clusters (int): Number of desired clusters (prototypes).
    - metric (str): Distance metric to use for clustering ('euclidean', 'manhattan', 'cosine').
    - num_prototypes (int): Number of representative examples to return for each cluster.

    Returns:
    - pd.DataFrame: DataFrame of representative examples for each cluster.
    """

    # Pivot data to create a consumption matrix
    consumption_matrix = data.pivot(index = 'PDE', columns = 'timestamp', values = 'QTD_CONSUMO_Adjusted').fillna(-1)

    # Standardize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(consumption_matrix)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters = num_clusters, random_state = 42)
    labels = kmeans.fit_predict(scaled_data)

    # Add the labels back to the original DataFrame
    consumption_matrix['Cluster'] = labels

    # Get representative examples for each cluster
    representative_examples = []
    for cluster in range(num_clusters):
        cluster_data = consumption_matrix[consumption_matrix['Cluster'] == cluster]

        # Get the indices of the representative examples
        # For simplicity, we can take the first few entries as examples
        if len(cluster_data) > num_prototypes:
            examples = cluster_data.sample(num_prototypes)  # Randomly select num_examples
        else:
            examples = cluster_data  # If fewer examples than required, take all

        representative_examples.append(examples)

    # Concatenate examples into a single DataFrame
    representative_df = pd.concat(representative_examples)

    return representative_df.reset_index()[['PDE', 'Cluster']]


def prototype_distance(data, metric = 'euclidean', num_clusters = 3, num_prototypes = 3):
    """
    Clusters consumers based on their consumption series using pairwise distances.

    Parameters:
    - data (pd.DataFrame): DataFrame containing consumption series with 'PDE' as a categorical column.
    - num_clusters (int): Number of desired clusters (prototypes).
    - metric (str): Distance metric to use for clustering ('euclidean', 'cityblock', 'cosine').
    - num_prototypes (int): Number of representative examples to return for each cluster.

    Returns:
    - pd.DataFrame: DataFrame of representative examples for each cluster.
    """

    # Pivot data to create a consumption matrix
    consumption_matrix = data.pivot(index = 'PDE', columns = 'timestamp', values = 'QTD_CONSUMO_Adjusted').fillna(-1)

    # Standardize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(consumption_matrix)

    # Compute pairwise distances
    distance_matrix = pdist(scaled_data, metric)

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method = 'ward')  # Using 'ward' linkage method
    labels = fcluster(linkage_matrix, num_clusters, criterion = 'maxclust')

    # Add the labels back to the original DataFrame
    consumption_matrix['Cluster'] = labels

    # Get representative examples for each cluster
    representative_examples = []
    for cluster in range(1, num_clusters + 1):  # Cluster labels start from 1
        cluster_data = consumption_matrix[consumption_matrix['Cluster'] == cluster]

        # Get the indices of the representative examples
        if len(cluster_data) > num_prototypes:
            examples = cluster_data.sample(num_prototypes)  # Randomly select num_prototypes
        else:
            examples = cluster_data  # If fewer examples than required, take all

        representative_examples.append(examples)

    # Concatenate examples into a single DataFrame
    representative_df = pd.concat(representative_examples)

    return representative_df.reset_index()[['PDE', 'Cluster']]


def cluster_consumers(data,
                      y_feat = 'QTD_CONSUMO_Adjusted', id_feat = 'PDE',
                      num_clusters = 2, threshold_cluster = 0.15,
                      method = 'distance', metric = 'cosine',
                      ):
    # Generate a color palette based on the number of clusters
    palette = sns.color_palette('tab10', num_clusters)

    data_resampled = filter_resample_TS(df = data, start_date = '2022-01-01',
                                        end_date = '2025-01-01', granularity = 'M')

    data_resampled_pde = filter_resample_TS(df = data, start_date = '2022-01-01',
                                            end_date = '2025-01-01', granularity = 'M', hue_by = id_feat)

    # cluster_kmeans = cluster_consumers(data_resampled_pde, num_clusters = num_clusters, num_examples = len(data_resampled_pde['PDE'].unique()))
    if method == 'distance':
        prototype_data = prototype_distance(data_resampled_pde, metric = metric, num_clusters = num_clusters,
                                            num_prototypes = len(data_resampled_pde[id_feat].unique()))
    else:
        prototype_data = prototype_Kmeans(data_resampled_pde, num_clusters = num_clusters,
                                          num_prototypes = len(data_resampled_pde[id_feat].unique()))

    # # TODO deixar para mais de 2 clusters:
    # cl_ratio = prototype_data['Cluster'].value_counts(normalize = True).reset_index()
    # if len(cl_ratio[cl_ratio['proportion'] < threshold_cluster]):
    #     data_clusters = data_resampled_pde.copy()
    #     data_clusters['Cluster'] = 0
    # else:

    data_clusters = pd.merge(data_resampled_pde, prototype_data, on = id_feat, how = 'left')
    data_clusters = data_clusters.sort_values(by = ['timestamp', 'Cluster'])

    cl_order = data_clusters[['Cluster', 'QTD_CONSUMO_Adjusted']].groupby('Cluster').mean().reset_index()
    max_index = cl_order['QTD_CONSUMO_Adjusted'].idxmax()
    if cl_order.loc[max_index, 'Cluster'] == 1:
        data_clusters['Cluster'] = data_clusters['Cluster'].replace({1: 2, 2: 1})

    data_clusters['Cl1_count'] = len(data_clusters[data_clusters['Cluster'] == 1]["PDE"].unique())
    data_clusters['Cl2_count'] = len(data_clusters[data_clusters['Cluster'] == 2]["PDE"].unique())

    # Create the subplots (2 rows, 1 column) with shared y-axis
    fig, axs = plt.subplots(2, 1, figsize = (16, 10), sharey = False)  # sharey=True for shared y-axis

    # First plot with kmeans clustering
    sns.lineplot(data = data_resampled, x = 'DATE', y = y_feat, ax = axs[0], color = 'gray', lw = 3,
                 ls = 'dashed', alpha = 0.6)
    sns.lineplot(data = data_clusters, x = 'DATE', y = y_feat, hue = 'Cluster',
                 palette = palette, ax = axs[0])
    axs[0].set_title('Cluster' + method)
    axs[0].legend(title = 'Cluster' + method, loc = 'upper right')

    # Second plot with distance-based clustering
    # Plot the instances with transparency
    for cluster_id in data_clusters['Cluster'].unique():
        cluster_data_id = data_clusters[data_clusters['Cluster'] == cluster_id]
        print(
            f'Tamanho cluster {cluster_id}:\t{len(cluster_data_id["PDE"].unique())} / {len(data_clusters["PDE"].unique())}')

        # Plot all individual instances with hue by 'PDE' and transparency
        sns.lineplot(
            data = cluster_data_id,
            x = 'DATE',
            y = y_feat,
            hue = id_feat,  # Hue by 'PDE' for each instance
            ax = axs[1],
            palette = [palette[cluster_id - 1]] * len(cluster_data_id[id_feat].unique()),
            # Assign same color for all PDEs in the cluster
            legend = False,  # Avoid legend to prevent clutter
            alpha = 0.15,  # Transparency for individual lines,
            lw = .3,
            errorbar = None
            )

        # Plot the aggregate line for the cluster
        sns.lineplot(
            data = cluster_data_id,
            x = 'DATE',
            y = y_feat,
            ax = axs[1],
            color = palette[cluster_id - 1],  # Same color for the cluster
            linewidth = 5,  # Make the aggregate line thicker for visibility
            label = f'Cluster {cluster_id}'  # Label the aggregate line
            )

    # Plot the hue with non-transparent lines for the cluster representatives
    # sns.lineplot(data=data_tgt_cluster, x=data_tgt_cluster.index, y='QTD_CONSUMO_Adjusted', hue='Cluster_dist', palette=palette, ax=axs[1])

    axs[1].set_title('Cluster' + method)
    axs[1].legend(title = 'Cluster' + method, loc = 'upper right')
    axs[1].get_legend().set_visible(False)

    # Display the plots
    plt.tight_layout()
    plt.show()

    return data_clusters, data_resampled

def dimensionality_reduction(data_raw, method = 'tSNE', n_components = 2, get_data = False, target_column = None,
                             random_state = 42):
    """
    Perform dimensionality reduction on categorical data using selected methods.

    Parameters:
    - data (DataFrame): Input DataFrame with categorical features.
    - method (str): Method to use for dimensionality reduction ('MCA', 'tSNE', 'UMAP', 'PCA').
    - n_components (int): Number of components to reduce to.
    - target_column (str): Column name for coloring the plot.
    - random_state (int): Seed for reproducibility.

    Returns:
    - reduced_data (DataFrame): DataFrame with reduced dimensions.
    """

    data = data_raw.copy()
    # If a target column is specified, get its values for coloring
    if target_column is not None and target_column in data.columns:
        target_values = data[target_column].values
        data = data.drop(columns = target_column)
        unique_categories = pd.Categorical(target_values).categories
    else:
        target_values = None
        unique_categories = None

    if method == 'tSNE':
        # t-Distributed Stochastic Neighbor Embedding
        tsne = TSNE(n_components = n_components, random_state = random_state)
        reduced_data = tsne.fit_transform(data)

    elif method == 'UMAP':
        # Uniform Manifold Approximation and Projection
        reducer = umap.UMAP(n_components = n_components, random_state = random_state)
        reduced_data = reducer.fit_transform(data)

    elif method == 'PCA':
        # Principal Component Analysis
        pca = PCA(n_components = n_components, random_state = random_state)
        reduced_data = pca.fit_transform(data)

    else:
        raise ValueError("Method not recognized. Choose 'tSNE', 'UMAP', or 'PCA'.")

    # Return reduced data as DataFrame
    reduced_df = pd.DataFrame(reduced_data, columns = [f'Component {i + 1}' for i in range(n_components)],
                              index = data.index)

    # df = data.reset_index(drop = True).copy()
    data['PC1'] = reduced_df['Component 1']
    data['PC2'] = reduced_df['Component 2']
    data[target_column] = target_values

    if not get_data:
        # Plotting the reduced data
        plt.figure(figsize = (8, 6))

        if target_values is not None:
            # Create a scatter plot colored by the target column
            scatter = plt.scatter(reduced_df.iloc[:, 0], reduced_df.iloc[:, 1],
                                  c = pd.Categorical(target_values).codes, alpha = 0.7,
                                  cmap = 'viridis', label = unique_categories)
            plt.colorbar(scatter, label = target_column)

            # Create a legend based on unique categories
            handles = [plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = scatter.cmap(scatter.norm(i)),
                                  markersize = 10)
                       for i in range(len(unique_categories))]
            plt.legend(handles, unique_categories, title = target_column)

        else:
            # Default scatter plot without coloring
            plt.scatter(reduced_df.iloc[:, 0], reduced_df.iloc[:, 1], alpha = 0.7, label = 'No target variable')

        plt.title(f'Dimensionality Reduction using {method}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid()
        plt.show()

    return data


def cluster_data(data, method = 'kmeans', n_clusters = 3, **kwargs):
    if method == 'kmeans':
        model = KMeans(n_clusters = n_clusters, **kwargs)
    elif method == 'dbscan':
        model = DBSCAN(**kwargs)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters = n_clusters, **kwargs)
    else:
        raise ValueError(f"Unknown method {method}")

    labels = model.fit_predict(data)
    return labels


def autoencoder_reduction(data, n_components = 2, hidden_layer_sizes = (64, 32)):
    # Define the Autoencoder using an MLPRegressor
    # n_components will be the number of neurons in the bottleneck layer (compressed representation)
    autoencoder = MLPRegressor(hidden_layer_sizes = (*hidden_layer_sizes, n_components, *hidden_layer_sizes[::-1]),
                               activation = 'relu', solver = 'adam', max_iter = 200, random_state = 42)

    # Create a pipeline to scale and reduce dimensionality
    pipeline = Pipeline(steps = [
        ('autoencoder', autoencoder)
        ])

    # Fit the model on the data
    pipeline.fit(data, data)

    # Use the bottleneck layer to obtain the reduced dimensionality representation
    # MLPRegressor's hidden_layer_sizes defines the layers, so we re-transform the data using these layers.
    reduced_data = autoencoder.predict(data)

    return reduced_data


def reduce_dimensions_and_plot(data, labels = None, method = 'pca', n_components = 2):
    if method == 'pca':
        reducer = PCA(n_components = n_components)
        reduced_data = reducer.fit_transform(data)
    elif method == 'tsne':
        reducer = TSNE(n_components = n_components, random_state = 42)
        reduced_data = reducer.fit_transform(data)
    elif method == 'autoencoder':
        reduced_data = autoencoder_reduction(data, n_components)
    else:
        raise ValueError(f"Unknown method {method}")

    # Plotting
    plt.figure(figsize = (8, 6))
    if n_components == 2:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = labels, cmap = 'viridis', s = 50)
        plt.title(f'{method.upper()} Dimensionality Reduction')
    elif n_components == 3:
        ax = plt.axes(projection = '3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c = labels, cmap = 'viridis', s = 50)
        plt.title(f'{method.upper()} Dimensionality Reduction (3D)')

    plt.colorbar()
    plt.show()

    return reduced_data


from sklearn.metrics import silhouette_samples, silhouette_score


def plot_silhouette(X, labels):
    # Get silhouette scores for each sample
    silhouette_vals = silhouette_samples(X, labels)

    # Calculate the mean silhouette score (overall score for the clustering)
    silhouette_avg = np.mean(silhouette_vals)

    n_clusters = len(np.unique(labels))
    y_lower, y_upper = 0, 0
    yticks = []

    # Create plot
    plt.figure(figsize = (10, 6))

    for i, cluster in enumerate(np.unique(labels)):
        # Aggregate silhouette scores for each cluster
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals)
        yticks.append((y_lower + y_upper) / 2)
        y_lower = y_upper

    # Draw a vertical line for the mean silhouette score
    plt.axvline(x = silhouette_avg, color = "red", linestyle = "--")
    plt.yticks(yticks, np.unique(labels) + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette Coefficient')
    plt.title('Silhouette Plot for the Clusters')
    plt.show()

    # Return the mean silhouette score
    return silhouette_avg


def run_clustering_experiment(X, cluster_methods, seeds, cluster_range, test_size = 0.2, normalize = 'minmax'):
    # List to store results
    results = []

    # Loop over dimension reduction methods, clustering methods, seeds, and number of clusters
    for seed in seeds:
        print('\n\n' + 20 * '**//')
        # Scale the data
        df_cluster_scaled = scale_data(X, y = None, normalize = normalize, split = False, random_state = seed)
        best_result = None  # To track the best clustering method for this dimension reduction method
        best_silhouette = -1  # Initialize best silhouette score
        for cluster in cluster_methods:
            for n_clusters in cluster_range:
                print(f'Clustering: {cluster}, Seed: {seed}, Clusters: {n_clusters}')

                # Apply clustering
                try:
                    cluster_labels = cluster_data(df_cluster_scaled, method = cluster, n_clusters = n_clusters)
                except Exception as e:
                    print(f"Clustering error: {e}")
                    continue

                # Calculate clustering quality metrics
                silhouette_avg = silhouette_score(df_cluster_scaled, cluster_labels)
                davies_bouldin = davies_bouldin_score(df_cluster_scaled, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(df_cluster_scaled, cluster_labels)

                # Save the results
                result = {
                    'clustering': cluster,
                    'seed': seed,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette_avg,
                    'davies_bouldin': davies_bouldin,
                    'calinski_harabasz': calinski_harabasz
                    }
                results.append(result)

                # Track the best result for this dimensionality reduction method
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_result = result

    # Convert results into a DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    results_df = results_df.groupby(['clustering', 'n_clusters']).mean().reset_index()

    # Sort by silhouette score or other metrics to find the best configuration
    sorted_results = results_df.sort_values(by = 'silhouette_score', ascending = False)

    return sorted_results


def select_best_clustering(data, davies_weight = 1.0, calinski_weight = .5, silhouette_weight = 2.0):
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Normalize each metric using sklearn's MinMaxScaler
    data[['davies_bouldin_normalized']] = scaler.fit_transform(data[['davies_bouldin']])
    data[['calinski_harabasz_normalized']] = scaler.fit_transform(data[['calinski_harabasz']])
    data[['silhouette_normalized']] = scaler.fit_transform(data[['silhouette_score']])

    # Invert Davies-Bouldin after normalization (since lower is better)
    data['davies_bouldin_normalized'] = 1 - data['davies_bouldin_normalized']

    # Calculate the weighted score based on normalized metrics
    data['weighted_score'] = (
            davies_weight * data['davies_bouldin_normalized'] +
            calinski_weight * data['calinski_harabasz_normalized'] +
            silhouette_weight * data['silhouette_normalized']
    )

    # Sort by the weighted score in descending order (higher score is better)
    sorted_data = data.sort_values(by = 'weighted_score', ascending = False)

    # Return the best result (highest score)
    return sorted_data.iloc[0]


def evaluate_feature_importance(X, tgt_feat):
    # Split the data into features and target (cluster labels)
    y = X[tgt_feat]
    X = X.drop(columns = [tgt_feat])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Fit RandomForestClassifier
    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)

    # Get feature importance
    feature_importances = model.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by = 'importance', ascending = False)

    return feature_importance_df


def plot_feature_importance(feature_importance_df):
    plt.figure(figsize = (10, 6))
    sns.barplot(x = 'importance', y = 'feature', data = feature_importance_df)
    plt.title('Feature Importance for Clustering')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()


def critical_investigation(df, cluster_col):
    """
    Perform a critical investigation of clustering results.

    Parameters:
    - df: DataFrame containing the features and cluster labels.
    - cluster_col: Name of the column that contains cluster labels.

    Returns:
    - None
    """
    # Ensure the cluster column is categorical
    df[cluster_col] = df[cluster_col].astype('category')

    # 1. Feature Importance using Random Forest
    X = df.drop(columns = [cluster_col])
    y = df[cluster_col]

    rf = RandomForestClassifier()
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by = 'Importance', ascending = False)

    # Plot Feature Importance
    plt.figure(figsize = (10, 6))
    sns.barplot(x = 'Importance', y = 'Feature', data = feature_importance_df)
    plt.title('Feature Importance from Random Forest')
    plt.show()

    # 2. Correlation Matrix
    plt.figure(figsize = (12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot = True, fmt = ".2f", cmap = 'coolwarm', square = True)
    plt.title('Correlation Matrix')
    plt.show()

    # 3. Box Plots for each feature by cluster
    features = X.columns
    for feature in features:
        plt.figure(figsize = (10, 6))
        sns.boxplot(x = cluster_col, y = feature, data = df)
        plt.title(f'Box Plot of {feature} by {cluster_col}')
        plt.show()

    # 4. PCA for Visualization
    pca = PCA(n_components = 2)
    principal_components = pca.fit_transform(X)
    pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
    pca_df[cluster_col] = y

    plt.figure(figsize = (10, 6))
    sns.scatterplot(x = 'PC1', y = 'PC2', hue = cluster_col, data = pca_df, palette = 'deep')
    plt.title('PCA of Clusters')
    plt.show()
#
#
# def run_clustering_experiment(X, dimension_reduction_methods, cluster_methods, seeds, cluster_range, test_size = 0.2,
#                               normalize = 'minmax'):
#     # List to store results
#     results = []
#
#     # Loop over dimension reduction methods, clustering methods, seeds, and number of clusters
#     for dim in dimension_reduction_methods:
#         print('\n\n' + 20 * '**//')
#
#         for cluster in cluster_methods:
#             for seed in seeds:
#                 for n_clusters in cluster_range:
#                     print(f'Dimension Reduction: {dim}, Clustering: {cluster}, Seed: {seed}, Clusters: {n_clusters}')
#
#                     # Scale the data
#                     df_cluster_scaled = scale_data(X, y = None, normalize = normalize, split = False,
#                                                    random_state = seed)
#
#                     # Apply clustering
#                     try:
#                         cluster_labels = cluster_data(df_cluster_scaled, method = cluster, n_clusters = n_clusters)
#                     except Exception as e:
#                         print(f"Clustering error: {e}")
#                         continue
#
#                     # Calculate clustering quality metrics
#                     silhouette_avg = silhouette_score(df_cluster_scaled, cluster_labels)
#                     davies_bouldin = davies_bouldin_score(df_cluster_scaled, cluster_labels)
#                     calinski_harabasz = calinski_harabasz_score(df_cluster_scaled, cluster_labels)
#
#                     # Save the results
#                     result = {
#                         'dim_reduction': dim,
#                         'clustering': cluster,
#                         'seed': seed,
#                         'n_clusters': n_clusters,
#                         'silhouette_score': silhouette_avg,
#                         'davies_bouldin': davies_bouldin,
#                         'calinski_harabasz': calinski_harabasz
#                         }
#                     results.append(result)
#
#     # Convert results into a DataFrame for easier analysis
#     results_df = pd.DataFrame(results)
#
#     # Calculate the mean of the metrics grouped by dimension reduction method, clustering method, and seed
#     grouped_results = results_df.groupby(['dim_reduction', 'clustering', 'seed']).agg({
#         'silhouette_score': 'mean',
#         'davies_bouldin': 'mean',
#         'calinski_harabasz': 'mean'
#         }).reset_index()
#
#     # Calculate the mean of all the metrics combined for ranking (you can adjust how to combine the metrics if necessary)
#     # grouped_results['mean_score'] = grouped_results[['silhouette_score', 'davies_bouldin', 'calinski_harabasz']].mean(axis=1)
#
#     # Sort by the mean score
#     best_result = grouped_results.sort_values(by = 'silhouette_score', ascending = False).iloc[0]
#
#     # Print the best result
#     print(f"\n\nBest result: {best_result}")
#
#     # Plot the best result
#     df_cluster_scaled = scale_data(X, y = None, normalize = normalize, split = False,
#                                    random_state = best_result['seed'])
#     cluster_labels = cluster_data(df_cluster_scaled, method = best_result['clustering'],
#                                   n_clusters = best_result['n_clusters'])
#     _ = reduce_dimensions_and_plot(df_cluster_scaled, labels = cluster_labels, method = best_result['dim_reduction'],
#                                    n_components = 2)
#     plot_silhouette(df_cluster_scaled, cluster_labels)
#
#     return grouped_results
