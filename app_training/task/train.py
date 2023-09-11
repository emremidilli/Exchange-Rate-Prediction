import lightning.pytorch as pl

from models import get_model

import os

import torch

from utils import filter_relevant_time_idx, get_converted_data, \
    get_timestamps, convert_to_time_series_dataset, filter_dataset, \
    get_dataloaders, get_training_args


if __name__ == '__main__':
    args = get_training_args()
    CONVERTED_PARENT_DIR = '../tsf-bin/01 - Converted Data'
    TRAINING_PARENT_DIR = '../tsf-bin/02 - Training Datasets'
    ARTIFACTS_PARENT_DIR = '../tsf-bin/04 - Artifacts'
    FORECAST_HORIZON = 120
    LOOKBACK_HORIZON = 480

    CHANNEL = input(f'Enter a channel name from {TRAINING_PARENT_DIR}: \n')

    TRAINING_DATA_DIR = os.path.join(TRAINING_PARENT_DIR, CHANNEL)
    CONVERTED_DATA_DIR = os.path.join(
        CONVERTED_PARENT_DIR,
        f'{CHANNEL}.csv')
    LOGS_DIR = os.path.join(ARTIFACTS_PARENT_DIR, CHANNEL, args.model_type)
    ARTIFACTS_DIR = os.path.join(
        ARTIFACTS_PARENT_DIR,
        CHANNEL,
        args.model_type,
        'saved_model',
        'model.pth'
        )

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

    ds_train, ds_test = convert_to_time_series_dataset(
        df_train=df_train,
        df_test=df_test,
        lookback_horizon=LOOKBACK_HORIZON,
        forecast_horizon=FORECAST_HORIZON,
        model_type=args.model_type
    )

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
        model_type=args.model_type,
        ds=ds_train)
    print(f"Number of parameters in network: {model.size()/1e3:.1f}k")

    trainer.fit(
        model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_test)

    if os.path.exists(os.path.dirname(ARTIFACTS_DIR)) is False:
        os.makedirs(os.path.dirname(ARTIFACTS_DIR))

    torch.save(
        model.state_dict(),
        ARTIFACTS_DIR)

    print('Training completed successfully.')
