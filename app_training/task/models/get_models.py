from pytorch_forecasting import NBeats, NHiTS, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


def get_model(model_type, ds):
    '''returns a model based on model_type'''
    if model_type == 'tft':
        return get_tft_model(ds)
    elif model_type == 'nbeats':
        return get_nbeats_model(ds)
    elif model_type == 'nhits':
        return get_nhits_model(ds)


def get_tft_model(ds):
    '''returns temporal fusion transformer model'''
    model = TemporalFusionTransformer.from_dataset(
        ds,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,
        optimizer='Adam',
        reduce_on_plateau_patience=4)

    return model


def get_nbeats_model(ds):
    '''returns an NBeats model'''

    model = NBeats.from_dataset(
        ds,
        learning_rate=1e-3,
        log_interval=10,
        log_val_interval=1,
        widths=[32, 512],
        optimizer='Adam',
        backcast_loss_ratio=1.0)

    return model


def get_nhits_model(ds):
    '''returns an NHiTS model.'''

    model = NHiTS.from_dataset(
        ds,
        learning_rate=1e-3,
        log_interval=10,
        log_val_interval=1,
        optimizer='Adam',
        backcast_loss_ratio=1.0)

    return model
