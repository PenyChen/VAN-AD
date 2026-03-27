
import pandas as pd
from typing import Union, Any, Optional, Dict
import numpy as np

def read_data(path: str, nrows=None) -> Union[pd.DataFrame, np.ndarray]:
    """
    Read the data file and return DataFrame.If the data is spatial-temporal format,

    return it as a numpy array; otherwise, return it as a Pandas DataFrame.

    :param path: The path to the data file.
    :return:  The content of the data file.
    """
    data = pd.read_csv(path)
    if is_st(data):
        return process_data_np(data, nrows)
    else:
        return process_data_df(data, nrows)
    
def is_st(data: pd.DataFrame) -> bool:
    """
    Checks if data of the CSV file are in spatial-temporal format.

    :param data: The series data.
    :return: Are all values in 'cols' column are in spatial-temporal format.
    """
    return data.shape[1] == 4 and "id" in data.columns


def process_data_np(df: pd.DataFrame, nrows=None) -> np.ndarray:
    """
    Convert spatial-temporal data from a DataFrame

    to a three-dimensional(time stamp,feature,sensor)  numpy array.

    :param df: Spatial-temporal data.
    :param nrows: Optional, number of rows to retain. Default is None, retaining all rows.
    :return: Three-dimensional(time stamp,feature,sensor) numpy array of the spatial temporal data.
    """
    pivot_df = df.pivot_table(index="date", columns=["id", "cols"], values="data")

    sensors = df["id"].unique()
    features = df["cols"].unique()
    pivot_df = pivot_df.reindex(
        columns=pd.MultiIndex.from_product([sensors, features]), fill_value=np.nan
    )

    data_np = pivot_df.to_numpy().reshape(len(pivot_df), len(sensors), len(features))
    data_np = np.transpose(data_np, (0, 2, 1))

    if nrows is not None:
        data_np = data_np[:nrows, :, :]

    return data_np


def process_data_df(data: pd.DataFrame, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.

    According to the provided file path, read the data file and return the corresponding DataFrame.

    :param data: Data frame to read.
    :return:  The DataFrame of the content of the data file.
    """
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df