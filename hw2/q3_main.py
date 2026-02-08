from data import sample_data
from loss import MSELoss, BCELoss, BCEFromLogitsLoss
from optimizer import SGD, SGDWithMomentum, Adam
from train import train_model
from utils import set_global_seed, count_binary_correct, count_binary_correct_from_logits, visualize

if __name__ == '__main__':
    set_global_seed()

    data = dict(zip(['x_train', 'y_train', 'x_val', 'y_val'], sample_data('XOR')))
    config = dict(
        training_config = dict(
            epochs=500,
            criterion=BCELoss,
        ),
        model_config = dict(
            layer_nodes=[2, 4, 1],
            optimizer=Adam(lr=2e-2),
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
            batch_size=200,
            shuffle=False,
        ),
    )

    model, history = train_model(**config)
    visualize(model, history, config['valset_config'], save_to='q3.jpg')
