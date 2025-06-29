# filename: processing.py
import pandas as pd
import numpy as np

def process_and_score_stocks(
    six_month_correlations,
    three_month_correlations,
    screener_data_df,
    source_ticker,
    min_nodes,
    max_nodes,
    threshold_percent
):
    """
    Processes stock correlation data for a specific source ticker.
    It filters for positive correlations, computes a dynamic impact score (gravitational_force),
    filters connections, and then calculates a final net gravitational force and the
    maximum potential force under ideal conditions.

    Args:
      six_month_correlations: The six-month spearman lagged correlation matrix.
      three_month_correlations: The three-month spearman lagged correlation matrix.
      screener_data_df: DataFrame with additional stock information.
      source_ticker: The ticker symbol for which to process data.
      min_nodes: Minimum number of correlated stocks to return.
      max_nodes: Maximum number of correlated stocks to return.
      threshold_percent: A percentage (0.0 to 1.0) of the maximum force to use as a filter.

    Returns:
      processed_data_df: A pandas DataFrame with processed data for visualization.
      source_data_df: A pandas DataFrame containing the net_gravitational_force,
                      max_potential_force, and gravitational_impact for the source ticker,
                      along with the source ticker's market cap influence and source_moon_radius.
    """
    # --- Data Unpivoting and Initial Setup ---
    # planett with the 6-month correlation data as the base
    correlation_df = six_month_correlations.rename_axis('source', axis=0)
    grouped_correlation_data = correlation_df.stack().reset_index()
    grouped_correlation_data.columns = ['source', 'target', 'six_month_spearman_correlation']

    grouped_correlation_data = grouped_correlation_data[
        (grouped_correlation_data['source'] != grouped_correlation_data['target']) &
        (grouped_correlation_data['target'] != source_ticker)
    ].copy()

    # --- Filter for the specific source ticker ---
    source_connections = grouped_correlation_data[grouped_correlation_data['source'] == source_ticker].copy()
    if source_connections.empty:
        print(f"No correlation data found for source ticker {source_ticker}.")
        # Return empty dataframes when no data is found
        return pd.DataFrame(), pd.DataFrame()

    # Add 3-month correlation data before filtering
    source_connections['three_month_spearman_correlation'] = source_connections.apply(
        lambda row: three_month_correlations.loc[row['source'], row['target']] if row['source'] in three_month_correlations.index and row['target'] in three_month_correlations.columns else 0, axis=1
    )

    # We only care about positively correlated stocks for this model in both 6 and 3 month periods
    positive_corr_group = source_connections[
        (source_connections['six_month_spearman_correlation'] > 0) &
        (source_connections['three_month_spearman_correlation'] > 0)
    ].copy()

    if positive_corr_group.empty:
        print(f"No positive correlations found for source ticker {source_ticker}.")
        # Return empty dataframes when no data is found
        return pd.DataFrame(), pd.DataFrame()

    # --- Enrich Data (before filtering) ---
    # Add market data
    screener_cols_to_add = ['code', 'market_capitalization', 'last_day_change']
    required_screener_cols = ['code', 'market_capitalization', 'last_day_change']
    if not all(col in screener_data_df.columns for col in required_screener_cols):
        missing = [col for col in required_screener_cols if col not in screener_data_df.columns]
        raise ValueError(f"screener_data_df is missing required columns: {missing}")

    screener_info = screener_data_df[screener_cols_to_add].rename(columns={'code': 'target'})
    positive_corr_group = pd.merge(positive_corr_group, screener_info, on='target', how='left')
    positive_corr_group.dropna(subset=['market_capitalization', 'last_day_change'], inplace=True)
    if positive_corr_group.empty:
        print(f"No valid connections after merging screener data for {source_ticker}.")
        # Return empty dataframes when no data is found
        return pd.DataFrame(), pd.DataFrame()


    # --- Calculate Dynamic Impact Score (Gravitational Force) ---
    epsilon = 1e-9 # Small value to avoid log(0) issues.
    # Weights for recency bias
    w_3m = 0.6
    w_6m = 0.4
    # "unified_correlation" is a weighted average of recent correlations.
    positive_corr_group['unified_correlation'] = (
        w_3m * positive_corr_group['three_month_spearman_correlation'] +
        w_6m * positive_corr_group['six_month_spearman_correlation']
    )

    # Calculate a market cap influence score scaled between 0 and 1 for target stocks.
    positive_corr_group['Market Cap'] = positive_corr_group['market_capitalization']

    # --- Calculate source ticker's market cap and log cap ---
    source_screener_info = screener_data_df[screener_data_df['code'] == source_ticker]
    source_market_cap = source_screener_info['market_capitalization'].iloc[0] if not source_screener_info.empty and 'market_capitalization' in source_screener_info.columns else epsilon
    source_log_cap = np.log(max(source_market_cap, epsilon))


    # Calculate log market caps for all relevant tickers (source and targets)
    all_market_caps = positive_corr_group['Market Cap'].tolist()
    all_market_caps.append(source_market_cap) # Include source market cap

    log_caps = np.log(pd.Series(all_market_caps).clip(lower=epsilon))

    min_log_cap, max_log_cap = log_caps.min(), log_caps.max()
    log_cap_range = max_log_cap - min_log_cap

    # Calculate market cap influence for target stocks
    if log_cap_range > 0:
        positive_corr_group['market_cap_influence'] = np.log(positive_corr_group['Market Cap'].clip(lower=epsilon))
    else:
        positive_corr_group['market_cap_influence'] = 20 # Neutral value if all caps are the same


    # The `gravitational_force` is a product of recent correlation strength and market influence.
    # Modified: Increased the influence of unified_correlation by multiplying by a factor
    correlation_weight_factor = 1.0 # Factor to increase the influence of unified_correlation
    positive_corr_group['gravitational_force'] = (
        (positive_corr_group['unified_correlation'] * correlation_weight_factor) * # Multiply unified_correlation by a factor
        positive_corr_group['market_cap_influence']
    )

    # --- Apply Filtering ---
    max_abs_force = positive_corr_group['gravitational_force'].abs().max()
    if pd.isna(max_abs_force) or max_abs_force == 0:
        # Return empty dataframes when no data is found
        return pd.DataFrame(), pd.DataFrame()

    force_threshold = max_abs_force * threshold_percent
    filtered_by_force_threshold = positive_corr_group[positive_corr_group['gravitational_force'].abs() >= force_threshold].copy()

    # Enforce min/max node constraints
    if len(filtered_by_force_threshold) < min_nodes:
        final_filtered_df = positive_corr_group.sort_values(by='gravitational_force', key=abs, ascending=False).head(min_nodes).copy()
    elif len(filtered_by_force_threshold) > max_nodes:
        final_filtered_df = filtered_by_force_threshold.sort_values(by='gravitational_force', key=abs, ascending=False).head(max_nodes).copy()
    else:
        final_filtered_df = filtered_by_force_threshold.copy()

    if final_filtered_df.empty:
        print(f"No connections remained for {source_ticker} after filtering.")
        # Return empty dataframes when no data is found
        return pd.DataFrame(), pd.DataFrame()

    # --- Calculate Final Net Force and Visualization Parameters ---
    final_filtered_df['Daily Change'] = final_filtered_df['last_day_change']

    final_filtered_df['signed_gravitational_force'] = final_filtered_df.apply(
        lambda row: row['gravitational_force'] if row['Daily Change'] >= 0 else -row['gravitational_force'],
        axis=1
    )

    net_gravitational_force = final_filtered_df['signed_gravitational_force'].sum()
    max_potential_force = final_filtered_df['market_cap_influence'].sum()

    # --- Calculate Visualization Parameters ---
    min_corr, max_corr = final_filtered_df['gravitational_force'].min(), final_filtered_df['gravitational_force'].max()
    corr_range = max_corr - min_corr if max_corr > min_corr else 1.0
    # MODIFIED: Reverse the scaling for Orbital Radius
    if corr_range > 0:
        final_filtered_df['Orbital Radius'] = 1 - ((final_filtered_df['gravitational_force'] - min_corr) / corr_range)
    else:
        final_filtered_df['Orbital Radius'] = 0.5 # Neutral value if all forces are the same

    # -----Calculate moon Radius------
    # Combine all market caps to find the true min and max for normalization
    all_caps = pd.concat([
        final_filtered_df['Market Cap'],
        pd.Series([source_market_cap]) # Make sure source_market_cap is a Series
    ], ignore_index=True)

    # Calculate the log, clipping to avoid errors with zero
    log_all_caps = np.log(all_caps.clip(lower=epsilon))

    # Find the min and max from the complete set of data
    min_log_cap = log_all_caps.min()
    max_log_cap = log_all_caps.max()
    log_cap_range = max_log_cap - min_log_cap

    # Now, apply the normalization ONLY to the DataFrame's data
    # using the min/max from the combined set
    if log_cap_range > 0:
        # We are calculating log on just the dataframe column now
        log_df_caps = np.log(final_filtered_df['Market Cap'].clip(lower=epsilon))
        final_filtered_df['moon Radius'] = (log_df_caps - min_log_cap) / log_cap_range
    else:
        # If all values are the same, assign a default radius
        final_filtered_df['moon Radius'] = 0.5

    # Calculate source_moon_radius using the same min/max log caps from the targets and source.
    if log_cap_range > 0:
        source_moon_radius = (source_log_cap - min_log_cap) / log_cap_range
    else:
        source_moon_radius = 0.5 # Neutral value if all caps are the same

    # --- Final Cleanup and Column Selection ---
    # "gravitational_percent" shows the relative % contribution of each stock.
    final_filtered_df['gravitational_percent'] = (final_filtered_df['signed_gravitational_force'] / final_filtered_df['gravitational_force'].sum()) * 100

    final_columns = [
        'source', 'target', 'Daily Change', 'six_month_spearman_correlation',
        'three_month_spearman_correlation', 'unified_correlation',
        'Orbital Radius', 'Market Cap', 'moon Radius', 'market_cap_influence',
        'gravitational_force', 'signed_gravitational_force', 'gravitational_percent'
    ]


    gravitational_impact = (net_gravitational_force / max_potential_force) * 100 if max_potential_force > 0 else 0

    # Use the same min_log_cap and log_cap_range from target stocks for scaling
    source_market_cap_influence = 20 if log_cap_range <= 0 else (source_log_cap)

    # Create source_data_df
    source_data_df = pd.DataFrame([{
        'ticker': source_ticker,
        'net_gravitational_force': net_gravitational_force,
        'max_potential_force': max_potential_force,
        'gravitational_impact': gravitational_impact,
        'source_market_cap_influence': source_market_cap_influence, # Add the source influence
        'source_moon_radius': source_moon_radius # Add the source moon radius
    }])


    for col in final_columns:
        if col not in final_filtered_df.columns:
            final_filtered_df[col] = np.nan

    processed_data_df = final_filtered_df[final_columns].copy()

    return processed_data_df, source_data_df
