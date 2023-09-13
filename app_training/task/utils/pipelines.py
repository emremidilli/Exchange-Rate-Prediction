import numpy as np

import os

import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet


def get_converted_data(dir):
    '''gets the dataframe that is converted in pytorch format.'''
    df = pd.read_csv(dir, sep=';')
    df.query('group_id == "ticker"', inplace=True)
    return df


def get_timestamps(dir):
    '''get timeidx that are used during training and testing'''
    ts_train = np.load(
        os.path.join(dir, 'ix_train.npy'))
    ts_test = np.load(
        os.path.join(dir, 'ix_test.npy'))

    return ts_train, ts_test


def filter_relevant_time_idx(
        df_to_filter,
        ts_to_filter,
        lookback_horizon,
        forecast_horizon):
    '''calculates all the timeidx of lookback and forecast horizons.'''
    all_ts = ts_to_filter
    for i in range(lookback_horizon):
        all_ts = np.concatenate((all_ts, ts_to_filter - (i+1)))

    for i in range(forecast_horizon):
        all_ts = np.concatenate((all_ts, ts_to_filter + i))

    all_ts = np.unique(all_ts)
    all_ts = np.sort(all_ts)

    return df_to_filter.query('time_idx in @all_ts')


def convert_to_time_series_dataset(
        df_train,
        df_test,
        lookback_horizon,
        forecast_horizon,
        model_type):
    '''convert to TimeSeriesDataset'''
    add_relative_time_idx = False
    static_categoricals = []
    if model_type == 'tft':
        add_relative_time_idx = True
        static_categoricals = ['group_id']
    elif model_type == 'nbeats' or model_type == 'nhits':
        add_relative_time_idx = False
        static_categoricals = []

    ds_train = TimeSeriesDataSet(
        data=df_train,
        time_idx='time_idx',
        target='value',
        group_ids=['group_id'],
        min_encoder_length=lookback_horizon,
        max_encoder_length=lookback_horizon,
        min_prediction_length=forecast_horizon,
        max_prediction_length=forecast_horizon,
        constant_fill_strategy={'value': -1},
        allow_missing_timesteps=True,
        time_varying_unknown_reals=['value'],
        static_categoricals=static_categoricals,
        add_relative_time_idx=add_relative_time_idx,
    )

    ds_test = TimeSeriesDataSet.from_parameters(
        data=df_test,
        parameters=ds_train.get_parameters()
    )

    return ds_train, ds_test


def filter_dataset(ds_to_filter, ts_to_filter):
    '''filters the dataset based on the timestamps'''
    ds_to_filter.filter(
        lambda x: (x.time_idx_first_prediction.isin(ts_to_filter)),
        copy=False)


def get_dataloaders(ds_train, ds_test):
    '''returns dataloaders from datasets.'''
    dl_train = ds_train.to_dataloader(
        train=True,
        batch_size=128,
        shuffle=False,
        num_workers=4)
    dl_test = ds_test.to_dataloader(
        train=False,
        batch_size=len(ds_test),
        shuffle=False,
        num_workers=4)

    return dl_train, dl_test
