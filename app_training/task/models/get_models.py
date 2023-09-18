from pytorch_forecasting import NBeats, NHiTS, TemporalFusionTransformer
from pytorch_forecasting.metrics import MASE


def get_model(model_type, ds, hidden_size):
    '''returns a model based on model_type'''
    if model_type == 'tft':
        return get_tft_model(ds, hidden_size)
    elif model_type == 'nbeats':
        return get_nbeats_model(ds, hidden_size)
    elif model_type == 'nhits':
        return get_nhits_model(ds, hidden_size)


def get_tft_model(ds, hidden_size):
    '''returns temporal fusion transformer model'''
    model = TemporalFusionTransformer.from_dataset(
        ds,
        learning_rate=1e-4,
        hidden_size=hidden_size,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=hidden_size,
        loss=MASE(),
        log_interval=10,
        optimizer='Adam',
        reduce_on_plateau_patience=100)

    return model


def get_nbeats_model(ds, hidden_size):
    '''returns an NBeats model'''

    model = NBeats.from_dataset(
        ds,
        num_blocks=[4, 4],
        learning_rate=1e-4,
        log_interval=10,
        log_val_interval=1,
        dropout=0.1,
        widths=[hidden_size, hidden_size, hidden_size, hidden_size],
        optimizer='Adam',
        loss=MASE(),
        backcast_loss_ratio=1.0,
        reduce_on_plateau_patience=100)

    return model


def get_nhits_model(ds, hidden_size):
    '''returns an NHiTS model.'''

    model = NHiTS.from_dataset(
        ds,
        learning_rate=1e-4,
        log_interval=10,
        log_val_interval=1,
        n_blocks=[1, 1, 1, 1],
        hidden_size=hidden_size,
        dropout=0.1,
        optimizer='Adam',
        loss=MASE(),
        backcast_loss_ratio=1.0,
        reduce_on_plateau_patience=100)

    return model
