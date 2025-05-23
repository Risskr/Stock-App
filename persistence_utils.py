import pickle
import pandas as pd

def save_price_history(price_history_dict: dict[str, pd.Series], filepath: str) -> None:
    """
    Saves the historical price data to a file using pickle.

    Args:
        price_history_dict: Dictionary where keys are stock tickers (strings)
                            and values are pandas Series (DatetimeIndex, float prices).
        filepath: The path to the file where the data will be saved.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(price_history_dict, f)

def load_price_history(filepath: str) -> dict[str, pd.Series] | None:
    """
    Loads historical price data from a file using pickle.

    Args:
        filepath: The path to the file from which to load the data.

    Returns:
        A dictionary containing the historical price data, or None
        if the file is not found.
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def save_pearson_sums(pearson_sums_dict: dict[tuple[str, str], dict[str, float]], filepath: str) -> None:
    """
    Saves the Pearson sums data to a file using pickle.

    Args:
        pearson_sums_dict: Nested dictionary storing Pearson components.
                           Outer keys: tuples of stock tickers ('STOCK_A', 'STOCK_B').
                           Inner keys: 'N', 'sum_A', 'sum_B', 'sum_A_sq', 'sum_B_sq', 'sum_AB'.
        filepath: The path to the file where the data will be saved.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(pearson_sums_dict, f)

def load_pearson_sums(filepath: str) -> dict[tuple[str, str], dict[str, float]] | None:
    """
    Loads Pearson sums data from a file using pickle.

    Args:
        filepath: The path to the file from which to load the data.

    Returns:
        A nested dictionary containing the Pearson sums data, or None
        if the file is not found.
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
