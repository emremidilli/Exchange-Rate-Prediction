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


def add_date_features(df_to_add, datetime_features, numpy_time_freq):
    '''
    Adds date features in the given dataframe.
    dataframe: pandas dataframe with time_idx in numpy unix format.
    datetime_features: list of date features
    numpy_time_freq: numpy frequency of unix format.
    '''

    df_new = df_to_add.copy()

    time_idx = df_to_add.loc[:, 'time_idx']
    timestamps = np.array(
        time_idx,
        dtype=f'datetime64[{numpy_time_freq}]')
    timestamps = pd.Series(timestamps, index=df_new.index)

    for feature in datetime_features:

        df_2 = pd.DataFrame(
            {
                feature:
                eval(f'timestamps.dt.{feature}.astype(str).astype("category")')
            })

        df_new = pd.concat([df_new, df_2], axis=1)

    return df_new


def convert_to_time_series_dataset(
        df_train,
        df_test,
        lookback_horizon,
        forecast_horizon,
        model_type,
        datetime_features):
    '''converts to TimeSeriesDataset'''
    add_relative_time_idx = False
    static_categoricals = []
    time_varying_known_categoricals = []
    if model_type == 'tft':
        add_relative_time_idx = True
        static_categoricals = ['group_id']
        time_varying_known_categoricals = datetime_features
    elif model_type == 'nbeats':
        add_relative_time_idx = False
        static_categoricals = []
        time_varying_known_categoricals = []
    elif model_type == 'nhits':
        add_relative_time_idx = False
        static_categoricals = []
        time_varying_known_categoricals = datetime_features

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
        time_varying_known_categoricals=time_varying_known_categoricals
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


def save_prediction(pred, dir_to_save):
    '''
    saves prediction as a numpy file.
    pred: prediction file returned by model.predict()
    dir_to_save: directory to save prediction
    '''
    np_pred = np.array(pred['prediction'].cpu())

    if os.path.exists(dir_to_save) is False:
        os.makedirs(dir_to_save)

    np.save(
        file=os.path.join(dir_to_save, 'pred'),
        arr=np_pred)
