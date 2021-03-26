import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import datetime
import os

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import  GridSearchCV, cross_validate, TimeSeriesSplit
from sklearn.metrics import make_scorer, accuracy_score

class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):

    df = reader()
    #df = clean_dataset(df)

    df['hr'] = df.hr.astype(str)

    for i in range(df['hr'].shape[0]):
        if len(df['hr'][i])<2:
            df['hr'][i] = "0"+ df['hr'][i]

    indices = lambda row: row['dteday'] + " "+ row['hr'] +":00:00"
    df['datetime'] = df.apply(indices, axis = 1 )
    df.datetime = pd.to_datetime(df.datetime, infer_datetime_format=True)
    df1 = df.reset_index().drop(['instant', 'index'],axis =1).set_index('datetime')
    df1 = df1.reindex(pd.date_range(df1.index[0],df1.index[-1], freq= '1H'))

    df1['has'] = df1.index.date.astype(str)
    df2 = df1.groupby('has')['season','holiday','weekday','workingday','weathersit','temp','hum','windspeed','casual','registered','cnt',"atemp"].median()
    
    indices = df1.index
    null_indices = df1.loc[df1.isnull().any(axis = 1)].index

    for i in null_indices:
        df1.loc[i,['yr','mnth','hr']] = (i.year, i.month, i.hour)
        df1.loc[i,['season']] = df2.season[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['holiday']] = df2.holiday[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['weekday']] = df2.weekday[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['workingday']] = df2.workingday[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['weathersit']] = df2.weathersit[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['temp']] = df2.temp[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['hum']] = df2.hum[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['windspeed']] = df2.windspeed[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['casual']] = df2.casual[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['registered']] = df2.registered[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['cnt']] = df2.cnt[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]
        df1.loc[i,['atemp']] = df2.atemp[df2.index.get_loc(df1.has[df1.index.get_loc(i)])]


    df1.yr[df1.yr == 2011] = 0
    df1.yr[df1.yr == 2012] = 1

    df3 = df1.drop(['dteday', 'has','casual','registered'],axis =1)
    df3.season = df3.season.astype(str)
    df3.holiday = df3.holiday.astype(str)
    df3.workingday = df3.workingday.astype(str)
    df3.weathersit = df3.weathersit.astype(str)
    df3.windspeed = df3.windspeed.astype(int)
    df3.temp = df3.temp.astype(int)
    df3.hum = df3.hum.astype(int)
    df3.yr = df3.yr.astype(int)


    feature_columns = ["yr", "mnth","hr","season","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"]
    target_column = "cnt"
    y = df3[target_column]
    X = df3[feature_columns]
    X_train, y_train =  X[X['yr']==0], y[X['yr']==0]
    X_test, y_test = X[X['yr']==1], y[X['yr']==1]
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}


# def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
#     cleaning_fn = _chain(
#         [
#             _fix_pool_quality,
#             _fix_misc_feature,
#             _fix_fireplace_quality,
#             _fix_garage_variables,
#             _fix_lot_frontage,
#             _fix_alley,
#             _fix_fence,
#             _fix_masvnr_variables,
#             _fix_electrical,
#             _fix_basement_variables,
#             _fix_unhandled_nulls,
#         ]
#     )
#     df = cleaning_fn(df)
#     return df


# def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
#     def helper(df):
#         for fn in functions:
#             df = fn(df)
#         return df

#     return helper


# def _fix_pool_quality(df):
#     num_total_nulls = df["PoolQC"].isna().sum()
#     num_nulls_when_poolarea_is_zero = df[df["PoolArea"] == 0]["PoolQC"].isna().sum()
#     assert num_nulls_when_poolarea_is_zero == num_total_nulls
#     num_nulls_when_poolarea_is_not_zero = df[df["PoolArea"] != 0]["PoolQC"].isna().sum()
#     assert num_nulls_when_poolarea_is_not_zero == 0
#     df["PoolQC"] = df["PoolQC"].fillna("NP")
#     return df


# def _fix_misc_feature(df):
#     num_total_nulls = df["MiscFeature"].isna().sum()
#     num_nulls_when_miscval_is_zero = df[df["MiscVal"] == 0]["MiscFeature"].isna().sum()
#     num_nulls_when_miscval_is_not_zero = (
#         df[df["MiscVal"] != 0]["MiscFeature"].isna().sum()
#     )
#     assert num_nulls_when_miscval_is_zero == num_total_nulls
#     assert num_nulls_when_miscval_is_not_zero == 0
#     df["MiscFeature"] = df["MiscFeature"].fillna("No MF")
#     return df


# def _fix_fireplace_quality(df):
#     num_total_nulls = df["FireplaceQu"].isna().sum()
#     num_nulls_when_fireplaces_is_zero = (
#         df[df["Fireplaces"] == 0]["FireplaceQu"].isna().sum()
#     )
#     num_nulls_when_fireplaces_is_not_zero = (
#         df[df["Fireplaces"] != 0]["FireplaceQu"].isna().sum()
#     )
#     assert num_nulls_when_fireplaces_is_zero == num_total_nulls
#     assert num_nulls_when_fireplaces_is_not_zero == 0
#     df["FireplaceQu"] = df["FireplaceQu"].fillna("No FP")
#     return df


# def _fix_garage_variables(df):
#     num_area_zeros = (df["GarageArea"] == 0).sum()
#     num_cars_zeros = (df["GarageCars"] == 0).sum()
#     num_both_zeros = ((df["GarageArea"] == 0) & (df["GarageCars"] == 0.0)).sum()
#     assert num_both_zeros == num_area_zeros == num_cars_zeros
#     for colname in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
#         num_total_nulls = df[colname].isna().sum()
#         num_nulls_when_area_and_cars_capacity_is_zero = (
#             df[(df["GarageArea"] == 0.0) & (df["GarageCars"] == 0.0)][colname]
#             .isna()
#             .sum()
#         )
#         num_nulls_when_area_and_cars_capacity_is_not_zero = (
#             df[(df["GarageArea"] != 0.0) & (df["GarageCars"] != 0.0)][colname]
#             .isna()
#             .sum()
#         )
#         assert num_total_nulls == num_nulls_when_area_and_cars_capacity_is_zero
#         assert num_nulls_when_area_and_cars_capacity_is_not_zero == 0
#         df[colname] = df[colname].fillna("No Ga")

#     num_total_nulls = df["GarageYrBlt"].isna().sum()
#     num_nulls_when_area_and_cars_is_zero = (
#         df[(df["GarageArea"] == 0.0) & (df["GarageCars"] == 0.0)]["GarageYrBlt"]
#         .isna()
#         .sum()
#     )
#     num_nulls_when_area_and_cars_is_not_zero = (
#         df[(df["GarageArea"] != 0.0) & (df["GarageCars"] != 0.0)]["GarageYrBlt"]
#         .isna()
#         .sum()
#     )
#     assert num_nulls_when_area_and_cars_is_zero == num_total_nulls
#     assert num_nulls_when_area_and_cars_is_not_zero == 0
#     df["GarageYrBlt"].where(
#         ~df["GarageYrBlt"].isna(), other=df["YrSold"] + 1, inplace=True
#     )

#     return df


# def _fix_lot_frontage(df):
#     assert (df["LotFrontage"] == 0).sum() == 0
#     df["LotFrontage"].fillna(0, inplace=True)
#     return df


# def _fix_alley(df):
#     df["Alley"].fillna("NA", inplace=True)
#     return df


# def _fix_fence(df):
#     df["Fence"].fillna("NF", inplace=True)
#     return df


# def _fix_masvnr_variables(df):
#     df = df.dropna(subset=["MasVnrType", "MasVnrArea"])
#     df = df[~((df["MasVnrType"] == "None") & (df["MasVnrArea"] != 0.0))]
#     return df


# def _fix_electrical(df):
#     df.dropna(subset=["Electrical"], inplace=True)
#     return df


# def _fix_basement_variables(df):
#     colnames = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
#     cond = ~(
#         df["BsmtQual"].isna()
#         & df["BsmtCond"].isna()
#         & df["BsmtExposure"].isna()
#         & df["BsmtFinType1"].isna()
#         & df["BsmtFinType2"].isna()
#     )
#     for c in colnames:
#         df[c].where(cond, other="NB", inplace=True)
#     return df


# def _fix_unhandled_nulls(df):
#     df.dropna(inplace=True)
#     return df
