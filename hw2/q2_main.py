from data import sample_data
from loss import MSELoss, BCELoss, BCEFromLogitsLoss
from optimizer import SGD, SGDWithMomentum, Adam
from train import train_model
from utils import set_global_seed, count_binary_correct, count_binary_correct_from_logits, visualize

if __name__ == '__main__':
    set_global_seed()

    data = dict(zip(['x_train', 'y_train', 'x_val', 'y_val'], sample_data('linear-separable')))
    config = dict(
        training_config = dict(
            epochs=1000,
            criterion=MSELoss,
        ),
        model_config = dict(
            layer_nodes=[2, 1, 1],
            optimizer=SGD(lr=2e-1),
            hidden_act='tanh',
            output_act='sigmoid'
        ),
        validation_config = dict(
            count_correct=count_binary_correct,
            mute=True,
        ),
        trainset_config = dict(
            data=data['x_train'],
            labels=data['y_train'],
            batch_size=100,
            shuffle=True,
        ),
        valset_config = dict(
            data=data['x_val'],
            labels=data['y_val'],
            batch_size=100,
            shuffle=False,
        ),
    )

    set_global_seed()
    model, history = train_model(**config)
    visualize(model, history, config['valset_config'], save_to='q2.jpg')
