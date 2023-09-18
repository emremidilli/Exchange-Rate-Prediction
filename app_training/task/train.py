import lightning.pytorch as pl

from models import get_model

import os

import torch

from utils import filter_relevant_time_idx, get_converted_data, \
    get_timestamps, convert_to_time_series_dataset, filter_dataset, \
    get_dataloaders, get_training_args, get_data_format_config, \
    add_date_features, save_prediction


if __name__ == '__main__':
    args = get_training_args()
    CONVERTED_PARENT_DIR = '../tsf-bin/01_converted_data'
    TRAINING_PARENT_DIR = '../tsf-bin/02_training_datasets'
    ARTIFACTS_PARENT_DIR = '../tsf-bin/04_artifacts'
    INFERENCE_PARANET_DIR = '../tsf-bin/05_artifacts'

    CHANNEL = args.channel
    MODEL_TYPE = args.model_type
    HIDDEN_SIZE = args.hidden_size

    TRAINING_DATA_DIR = os.path.join(TRAINING_PARENT_DIR, CHANNEL)
    CONVERTED_DATA_DIR = os.path.join(
        CONVERTED_PARENT_DIR,
        f'{CHANNEL}.csv')
    LOGS_DIR = os.path.join(ARTIFACTS_PARENT_DIR, CHANNEL, MODEL_TYPE)
    ARTIFACTS_DIR = os.path.join(
        LOGS_DIR,
        'saved_model',
        'model.pth')

    config = get_data_format_config(
        folder_path=TRAINING_DATA_DIR)

    FORECAST_HORIZON = config['forecast_horizon']
    LOOKBACK_HORIZON = FORECAST_HORIZON * config['lookback_coefficient']
    DATETIME_FEATURES = config['datetime_features']
    RAW_FREQ = config['raw_frequency']

    ts_train, ts_test = get_timestamps(TRAINING_DATA_DIR)
    df_data = get_converted_data(CONVERTED_DATA_DIR)

    df_train = filter_relevant_time_idx(
        df_to_filter=df_data,
        ts_to_filter=ts_train,
        lookback_horizon=LOOKBACK_HORIZON,
        forecast_horizon=FORECAST_HORIZON)

    df_test = filter_relevant_time_idx(
        df_to_filter=df_data,
        ts_to_filter=ts_test,
        lookback_horizon=LOOKBACK_HORIZON,
        forecast_horizon=FORECAST_HORIZON)

    df_train = add_date_features(
        df_to_add=df_train,
        datetime_features=DATETIME_FEATURES,
        numpy_time_freq=RAW_FREQ)

    df_test = add_date_features(
        df_to_add=df_test,
        datetime_features=DATETIME_FEATURES,
        numpy_time_freq=RAW_FREQ)

    ds_train, ds_test = convert_to_time_series_dataset(
        df_train=df_train,
        df_test=df_test,
        lookback_horizon=LOOKBACK_HORIZON,
        forecast_horizon=FORECAST_HORIZON,
        model_type=MODEL_TYPE,
        datetime_features=DATETIME_FEATURES)

    filter_dataset(
        ds_to_filter=ds_train,
        ts_to_filter=ts_train)

    filter_dataset(
        ds_to_filter=ds_test,
        ts_to_filter=ts_test)

    dl_train, dl_test = get_dataloaders(
        ds_train=ds_train,
        ds_test=ds_test)

    trainer = pl.Trainer(
        max_epochs=args.nr_of_epochs,
        enable_model_summary=True,
        accelerator='gpu',
        gradient_clip_val=0.1,
        default_root_dir=LOGS_DIR
    )

    model = get_model(
        model_type=MODEL_TYPE,
        ds=ds_train,
        hidden_size=HIDDEN_SIZE)
    print(f"Number of parameters in network: {model.size()/1e3:.1f}k")

    trainer.fit(
        model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_test)

    pred = model.predict(dl_test, mode='raw')
    save_prediction(
        pred=pred,
        dir_to_save=os.path.join(
            INFERENCE_PARANET_DIR,
            CHANNEL,
            MODEL_TYPE))

    if os.path.exists(os.path.dirname(ARTIFACTS_DIR)) is False:
        os.makedirs(os.path.dirname(ARTIFACTS_DIR))

    torch.save(
        model.state_dict(),
        ARTIFACTS_DIR)
