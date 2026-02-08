import numpy as np
from copy import deepcopy
from data import sample_data
from loss import MSELoss, BCELoss, BCEFromLogitsLoss
from optimizer import SGD, SGDWithMomentum, Adam
from train import train_model
from utils import set_global_seed, count_binary_correct, count_binary_correct_from_logits, visualize

def xor_engineering(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, X[:, 0:1] * X[:, 1:2]], axis=1)

def swiss_engineering(X: np.ndarray) -> np.ndarray:
    X1 = X[:, 0:1]
    X2 = X[:, 1:2]
    return np.concatenate([X, X1**2, X2**2, X1*X2], axis=1)

if __name__ == '__main__':
    set_global_seed()

    x_train, y_train, x_val, y_val = sample_data('XOR')
    xor_configs = dict(
        training_config = dict(
            epochs=2000,
            criterion=BCEFromLogitsLoss,
        ),
        model_config = dict(
            layer_nodes=[3, 1],
            optimizer=Adam(lr=1e-1),
            hidden_act='relu',
        ),
        validation_config = dict(
            count_correct=count_binary_correct_from_logits,
            mute=True,
        ),
        trainset_config = dict(
            data=xor_engineering(x_train),
            labels=y_train,
            batch_size=64,
            shuffle=True,
        ),
        valset_config = dict(
            data=xor_engineering(x_val),
            labels=y_val,
            batch_size=200,
            shuffle=False,
        ),
    )

    set_global_seed()
    xor_model, xor_history = train_model(**xor_configs)
    visualize(xor_model, xor_history, xor_configs['valset_config'], pred_fn=lambda model, x: model.forward(xor_engineering(x)), save_to='q7_xor.jpg')

    set_global_seed()
    x_train, y_train, x_val, y_val = sample_data('swiss-roll')
    swiss_configs = dict(
        training_config = dict(
            epochs=3000,
            criterion=BCEFromLogitsLoss,
        ),
        model_config = dict(
            layer_nodes=[5, 10, 1],
            optimizer=Adam(lr=2e-2),
            hidden_act='relu',
        ),
        validation_config = dict(
            count_correct=count_binary_correct_from_logits,
            mute=True,
        ),
        trainset_config = dict(
            data=swiss_engineering(x_train),
            labels=y_train,
            batch_size=64,
            shuffle=True,
        ),
        valset_config = dict(
            data=swiss_engineering(x_val),
            labels=y_val,
            batch_size=200,
            shuffle=False,
        ),
    )

    set_global_seed()
    swiss_model, swiss_history = train_model(**swiss_configs)
    visualize(swiss_model, swiss_history, swiss_configs['valset_config'], pred_fn=lambda model, x: model.forward(swiss_engineering(x)), save_to='q7_swiss.jpg')