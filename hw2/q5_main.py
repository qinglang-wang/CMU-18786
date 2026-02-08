from copy import deepcopy
from data import sample_data
from loss import MSELoss, BCELoss, BCEFromLogitsLoss
from optimizer import SGD, SGDWithMomentum, Adam
from train import train_model
from utils import set_global_seed, count_binary_correct, count_binary_correct_from_logits, visualize

if __name__ == '__main__':
    set_global_seed()

    x_train, y_train, x_val, y_val = sample_data('sinusoid')
    vanilla_config = dict(
        training_config = dict(
            epochs=9000,
            criterion=BCEFromLogitsLoss,
        ),
        model_config = dict(
            layer_nodes=[2, 32, 16, 16, 1],
            optimizer=SGD(lr=2e-2),
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

    momentum_config = deepcopy(vanilla_config)
    momentum_config['model_config']['optimizer'] = SGDWithMomentum(1e-2)

    adam_config = deepcopy(vanilla_config)
    adam_config['model_config']['optimizer'] = Adam(2e-2)

    set_global_seed()
    vanilla_model, vanilla_history = train_model(**vanilla_config)
    visualize(vanilla_model, vanilla_history, vanilla_config['valset_config'], save_to='q5_vanilla.jpg')

    set_global_seed()
    momentum_model, momentum_history = train_model(**momentum_config)
    visualize(momentum_model, momentum_history, momentum_config['valset_config'], save_to='q5_momentum.jpg')

    set_global_seed()
    adam_model, adam_history = train_model(**adam_config)
    visualize(adam_model, adam_history, adam_config['valset_config'], save_to='q5_adam.jpg')