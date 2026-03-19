# utils package

import pandas as pd


def safe_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract column from df as pd.Series, handling MultiIndex columns safely."""
    data = df[col]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data
