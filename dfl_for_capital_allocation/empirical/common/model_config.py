"""Shared neural-network training configuration for empirical notebooks."""

HIDDEN_DIM = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
EPOCHS = 500
PATIENCE = 75
VALIDATION_FRACTION = 0.2
SEED = 42
ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)


def make_model_config(model_config_cls):
    return model_config_cls(hidden_dim=HIDDEN_DIM, lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
                            epochs=EPOCHS, patience=PATIENCE,
                            val_frac=VALIDATION_FRACTION, seed=SEED)
