from copy import deepcopy
from data import sample_data
from loss import MSELoss, BCELoss, BCEFromLogitsLoss
from optimizer import SGD, SGDWithMomentum, Adam
from train import train_model
from utils import set_global_seed, count_binary_correct, count_binary_correct_from_logits, visualize

if __name__ == '__main__':
    set_global_seed()

    x_train, y_train, x_val, y_val = sample_data('circle')
    regressor_config = dict(
        training_config = dict(
            epochs=300,
            criterion=MSELoss,
        ),
        model_config = dict(
            layer_nodes=[2, 10, 1],
            optimizer=Adam(lr=1e-2),
            hidden_act='relu',
        ),
        validation_config = dict(
            count_correct=count_binary_correct,
            mute=True,
        ),
        trainset_config = dict(
            data=x_train,
            labels=y_train,
            batch_size=100,
            shuffle=True,
        ),
        valset_config = dict(
            data=x_val,
            labels=y_val,
            batch_size=200,
            shuffle=False,
        ),
    )

    classifier_config = deepcopy(regressor_config)
    classifier_config['training_config']['criterion'] = BCEFromLogitsLoss
    classifier_config['validation_config']['count_correct'] = count_binary_correct_from_logits

    set_global_seed()
    regressor_model, regressor_history = train_model(**regressor_config)
    visualize(regressor_model, regressor_history, regressor_config['valset_config'], save_to='q4_regressor.jpg')

    set_global_seed()
    classifier_model, classifier_history = train_model(**classifier_config)
    visualize(classifier_model, classifier_history, classifier_config['valset_config'], save_to='q4_classifier.jpg')