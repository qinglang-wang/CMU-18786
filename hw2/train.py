import numpy as np
from typing import Dict
from network import MLP
from data import DataLoader
from loss import Loss
from utils import print_config

def train_model(model_config: Dict, training_config: Dict, validation_config: Dict, trainset_config: Dict, valset_config: Dict):
    print(f"Start training with configs:")
    print_config('model_config', model_config)
    print_config('training_config', training_config)
    print_config('trainset_config', trainset_config)
    print_config('validation_config', validation_config)
    print_config('valset_config', valset_config)
    
    model = MLP(**model_config)
    print(model)
    
    train_loader = DataLoader(**trainset_config)
    val_loader = DataLoader(**valset_config)
    criterion: Loss = training_config['criterion']()

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(training_config['epochs']):
        train_epoch_loss = 0
        for X, y in train_loader:
            out = model.forward(X)

            model.zero_grad()
            loss = criterion.forward(out, y)
            model.backward(criterion.backward())
            model.step()

            train_epoch_loss += loss

        history['train_loss'].append(train_epoch_loss / len(train_loader))
    
        val_epoch_loss = 0
        val_correct = 0
        for X, y in val_loader:
            out = model.forward(X)
            loss = criterion.forward(out, y)

            val_epoch_loss += loss
            val_correct += validation_config['count_correct'](out, y)

        history['val_loss'].append(val_epoch_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_loader.n_samples)
        if not validation_config.get('mute', False):
            print(f"Epoch {f'{epoch}'.zfill(len(str(training_config['epochs'])))}/{training_config['epochs']} - Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {history['val_acc'][-1]:.4f}")
    print(f"Training done, with highest validation accuracy of {np.max(history['val_acc'])}.")
    return model, history
