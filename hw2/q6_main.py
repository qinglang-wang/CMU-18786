from copy import deepcopy
from data import sample_data
from loss import MSELoss, BCELoss, BCEFromLogitsLoss
from optimizer import SGD, SGDWithMomentum, Adam
from train import train_model
from utils import set_global_seed, count_binary_correct, count_binary_correct_from_logits, visualize

if __name__ == '__main__':
    set_global_seed()

    x_train, y_train, x_val, y_val = sample_data('swiss-roll')
    configs = dict(
        training_config = dict(
            epochs=2000,
            criterion=BCEFromLogitsLoss,
        ),
        model_config = dict(
            layer_nodes=[2, 32, 16, 16, 1],
            optimizer=Adam(lr=5e-3),
            hidden_act='relu',
        ),
        validation_config = dict(
            count_correct=count_binary_correct_from_logits,
            mute=True,
        ),
        trainset_config = dict(
            data=x_train,
            labels=y_train,
            batch_size=64,
            shuffle=True,
        ),
        valset_config = dict(
            data=x_val,
            labels=y_val,
            batch_size=200,
            shuffle=False,
        ),
    )

    set_global_seed()
    model, history = train_model(**configs)
    visualize(model, history, configs['valset_config'], save_to='q6.jpg')