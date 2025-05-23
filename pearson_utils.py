import pandas as pd
from persistence_utils import save_price_history, save_pearson_sums, load_price_history, load_pearson_sums
import numpy as np # Added for sqrt and nan

def Initialize_Pearson_Sums(historical_data_df: pd.DataFrame, price_history_filepath: str, sums_filepath: str) -> None:
    """
    Initializes and saves Pearson sums (with 1-day lag for the second stock in pair)
    and original price history from historical stock data.

    The historical_data_df is expected to cover exactly the 1-year period for initialization.
    For a pair (ticker_A, ticker_B), the sums calculated correspond to correlating
    ticker_A's price at day 'd' with ticker_B's price at day 'd-1'.

    Args:
        historical_data_df: A pandas DataFrame with a MultiIndex ('ticker', 'date')
                            where 'date' is a pandas Timestamp. It must contain a 'close'
                            column with daily closing prices. This DataFrame should represent
                            exactly the 1-year data to be initialized.
        price_history_filepath: String, path to save the processed (original, non-lagged) price history.
        sums_filepath: String, path to save the calculated lagged Pearson sums.
    """
    price_history_to_save = {}
    pearson_sums = {}

    # 1. Populate price_history_to_save with original, non-lagged series
    unique_tickers = historical_data_df.index.get_level_values('ticker').unique()
    for ticker in unique_tickers:
        series = historical_data_df.loc[ticker]['close'].copy()
        if isinstance(series.index, pd.MultiIndex):
            series.index = series.index.get_level_values('date')
        price_history_to_save[ticker] = series

    # Save the original price history
    save_price_history(price_history_to_save, price_history_filepath)

    # 2. Calculate pairwise lagged sums
    tickers_list = list(price_history_to_save.keys())
    for i in range(len(tickers_list)):
        for j in range(len(tickers_list)): # Iterate through all, including i == j, though sums for (A,A_lagged) might be less common
            ticker_A = tickers_list[i]
            ticker_B = tickers_list[j]
            
            # For pearon_sums[(ticker_A, ticker_B)], we correlate A with B_lagged
            series_A_actual = price_history_to_save[ticker_A]
            series_B_for_lagging = price_history_to_save[ticker_B]
            series_B_lagged = series_B_for_lagging.shift(1)

            df_aligned_A_Blagged = pd.DataFrame({'A_actual': series_A_actual, 'B_lagged': series_B_lagged}).dropna()
            
            N_A_Blagged = len(df_aligned_A_Blagged)

            if N_A_Blagged == 0:
                sum_A_val, sum_A_sq_val, sum_B_val, sum_B_sq_val, sum_AB_val = 0,0,0,0,0
            else:
                sum_A_val = df_aligned_A_Blagged['A_actual'].sum()
                sum_A_sq_val = (df_aligned_A_Blagged['A_actual'] ** 2).sum()
                sum_B_val = df_aligned_A_Blagged['B_lagged'].sum() # Sum of B's lagged prices
                sum_B_sq_val = (df_aligned_A_Blagged['B_lagged'] ** 2).sum() # Sum of B's lagged prices squared
                sum_AB_val = (df_aligned_A_Blagged['A_actual'] * df_aligned_A_Blagged['B_lagged']).sum()
            
            pearson_sums[(ticker_A, ticker_B)] = {
                'N': N_A_Blagged,
                'sum_A': sum_A_val,         # Sum of ticker_A's actual prices
                'sum_A_sq': sum_A_sq_val,   # Sum of ticker_A's actual prices squared
                'sum_B': sum_B_val,         # Sum of ticker_B's 1-day lagged prices
                'sum_B_sq': sum_B_sq_val,   # Sum of ticker_B's 1-day lagged prices squared
                'sum_AB': sum_AB_val        # Sum of (ticker_A actual * ticker_B lagged)
            }
            
            # Note: The above loop structure calculates for (A,B) where B is lagged.
            # If a user requests correlation for (B,A) from Calculate_Incremental_Pearson,
            # and Calculate_Incremental_Pearson assumes the second element of the pair key is the one whose sums are for lagged data,
            # then the current structure is correct. The sums stored for key (B,A) will mean B's actual prices vs A's lagged prices.

    save_pearson_sums(pearson_sums, sums_filepath)

def Update_Pearson_Sums(new_daily_data_df: pd.DataFrame, price_history_filepath: str, sums_filepath: str) -> dict | None:
    """
    Updates Pearson sums and price history incrementally with new daily stock data.
    IMPORTANT: This function currently recalculates sums based on NON-LAGGED data.
    It would need significant modification to correctly update based on 1-day lag logic
    similar to the modified Initialize_Pearson_Sums.

    Args:
        new_daily_data_df: DataFrame with new daily closing prices.
                           Index: MultiIndex ('ticker', 'date'), Columns: ['close'].
                           'date' should be pandas Timestamp.
        price_history_filepath: Path to the existing stored price history file.
        sums_filepath: Path to the existing stored Pearson sums file.

    Returns:
        The updated pearson_sums dictionary, or None if new_daily_data_df is empty
        and no processing occurs.

    Raises:
        FileNotFoundError: If price_history_filepath or sums_filepath is not found.
    """
    price_history = load_price_history(price_history_filepath)
    if price_history is None:
        raise FileNotFoundError(f"Price history file not found: {price_history_filepath}. Run Initialize_Pearson_Sums first.")

    pearson_sums_loaded = load_pearson_sums(sums_filepath) # Load existing sums
    if pearson_sums_loaded is None:
        raise FileNotFoundError(f"Pearson sums file not found: {sums_filepath}. Run Initialize_Pearson_Sums first.")

    if new_daily_data_df.empty:
        # print("No new daily data provided. Returning current sums from file.")
        return pearson_sums_loaded 

    latest_new_date = new_daily_data_df.index.get_level_values('date').max()
    one_year_ago_from_latest = latest_new_date - pd.DateOffset(years=1)
    
    new_prices_by_ticker = {
        ticker: df.droplevel('ticker')['close'] 
        for ticker, df in new_daily_data_df.groupby(level='ticker')
    }

    all_involved_tickers = set(price_history.keys()) | set(new_prices_by_ticker.keys())

    for ticker in all_involved_tickers:
        current_series_for_ticker = price_history.get(ticker, pd.Series(dtype='float64', index=pd.to_datetime([])))
        
        if ticker in new_prices_by_ticker:
            new_series = new_prices_by_ticker[ticker]
            if not current_series_for_ticker.empty:
                new_series_filtered = new_series[new_series.index > current_series_for_ticker.index.max()]
            else:
                new_series_filtered = new_series
            
            current_series_for_ticker = pd.concat([current_series_for_ticker, new_series_filtered]).sort_index()
            current_series_for_ticker = current_series_for_ticker[~current_series_for_ticker.index.duplicated(keep='last')]

        current_series_for_ticker = current_series_for_ticker[current_series_for_ticker.index >= one_year_ago_from_latest]
        price_history[ticker] = current_series_for_ticker
        if current_series_for_ticker.empty: 
            del price_history[ticker]
    
    # --- Recalculation of Pearson Sums with 1-day lag for Stock B ---
    updated_pearson_sums = {}
    tickers_list = sorted(list(price_history.keys()))

    for i in range(len(tickers_list)):
        for j in range(len(tickers_list)): # Iterate all pairs for lagged sums
            ticker_A = tickers_list[i]
            ticker_B = tickers_list[j]

            series_A_actual = price_history.get(ticker_A)
            series_B_for_lagging = price_history.get(ticker_B)

            if series_A_actual is None or series_B_for_lagging is None or series_A_actual.empty or series_B_for_lagging.empty:
                # Ensure an entry for the pair if one series is empty after windowing
                updated_pearson_sums[(ticker_A, ticker_B)] = {'N': 0, 'sum_A': 0, 'sum_A_sq': 0, 'sum_B': 0, 'sum_B_sq': 0, 'sum_AB': 0}
                continue

            series_B_lagged = series_B_for_lagging.shift(1)
            df_aligned_A_Blagged = pd.DataFrame({'A_actual': series_A_actual, 'B_lagged': series_B_lagged}).dropna()
            
            N_A_Blagged = len(df_aligned_A_Blagged)

            if N_A_Blagged == 0:
                sum_A_val, sum_A_sq_val, sum_B_val, sum_B_sq_val, sum_AB_val = 0,0,0,0,0
            else:
                sum_A_val = df_aligned_A_Blagged['A_actual'].sum()
                sum_A_sq_val = (df_aligned_A_Blagged['A_actual'] ** 2).sum()
                sum_B_val = df_aligned_A_Blagged['B_lagged'].sum()
                sum_B_sq_val = (df_aligned_A_Blagged['B_lagged'] ** 2).sum()
                sum_AB_val = (df_aligned_A_Blagged['A_actual'] * df_aligned_A_Blagged['B_lagged']).sum()
            
            updated_pearson_sums[(ticker_A, ticker_B)] = {
                'N': N_A_Blagged, 'sum_A': sum_A_val, 'sum_A_sq': sum_A_sq_val,
                'sum_B': sum_B_val, 'sum_B_sq': sum_B_sq_val, 'sum_AB': sum_AB_val
            }

    save_price_history(price_history, price_history_filepath)
    save_pearson_sums(updated_pearson_sums, sums_filepath)
    return updated_pearson_sums


def Calculate_Incremental_Pearson(pearson_sums: dict[tuple[str, str], dict[str, float]]) -> dict[tuple[str, str], float]:
    """
    Calculates Pearson correlation coefficients from pre-computed sum components.
    It assumes that for a pair (A,B) in pearson_sums, the components for 'B' 
    (sum_B, sum_B_sq) are for the 1-day lagged series of B, and sum_A, sum_A_sq are for
    the non-lagged series of A. sum_AB is sum(A_current * B_lagged).

    Args:
        pearson_sums: A dictionary where keys are tuples of stock tickers (e.g., ('AAPL', 'MSFT'))
                      and values are dictionaries containing the sum components:
                      {'N': count, 'sum_A': sum_X, 'sum_A_sq': sum_X_sq,
                       'sum_B': sum_Y_lagged, 'sum_B_sq': sum_Y_sq_lagged, 'sum_AB': sum_X_Y_lagged}.

    Returns:
        A dictionary where keys are tuples of stock tickers and values are their
        Pearson correlation coefficient (float), calculated for A vs B_lagged.
        Returns np.nan for pairs where correlation cannot be computed.
    """
    correlations = {}
    for pair, sums_components in pearson_sums.items():
        N = sums_components.get('N', 0)
        sum_A = sums_components.get('sum_A', 0) # Corresponds to Stock A (current)
        sum_A_sq = sums_components.get('sum_A_sq', 0) # Corresponds to Stock A (current)
        sum_B_lagged = sums_components.get('sum_B', 0) # Corresponds to Stock B (lagged)
        sum_B_sq_lagged = sums_components.get('sum_B_sq', 0) # Corresponds to Stock B (lagged)
        sum_AB_lagged = sums_components.get('sum_AB', 0) # Corresponds to sum(A_current * B_lagged)

        if N == 0:
            correlations[pair] = np.nan
            continue

        numerator = (N * sum_AB_lagged) - (sum_A * sum_B_lagged)
        term_A_sq = (N * sum_A_sq) - (sum_A ** 2)
        term_B_lagged_sq = (N * sum_B_sq_lagged) - (sum_B_lagged ** 2)

        if term_A_sq <= 0 or term_B_lagged_sq <= 0:
            correlations[pair] = np.nan 
            continue
        
        denominator = np.sqrt(term_A_sq * term_B_lagged_sq)

        if denominator == 0:
            correlations[pair] = np.nan
        else:
            r = numerator / denominator
            correlations[pair] = np.clip(r, -1.0, 1.0)
            
    return correlations
