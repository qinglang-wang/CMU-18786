import torch
import torch.nn as nn
from data import get_cifar100_loaders
from models import ResNet18
from trainer import train_model
from utils import set_seed, plot_loss, plot_accuracy, visualize_predictions

torch.set_float32_matmul_precision('high')

def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = get_cifar100_loaders(batch_size=512, augment=True, num_workers=4)

    print("Experiment: Training ResNet18 from Scratch")
    model = ResNet18(num_classes=100)
    config = dict(
        epochs=100,
        optimizer='sgd',
        lr=3e-1,
        momentum=0.9,
        weight_decay=5e-4,
        scheduler='cosine',
        criterion=nn.CrossEntropyLoss,
        device=device,
        save_path='results/q2_resnet18_best.pth',
    )
    history = train_model(model, train_loader, val_loader, config)
    plot_loss(history, title='ResNet18', save_to='results/q2_resnet18_loss.png')
    plot_accuracy(history, title='ResNet18', save_to='results/q2_resnet18_acc.png')

    model.load_state_dict(torch.load('results/q2_resnet18_best.pth', weights_only=True))
    model = model.to(device)
    vis_indices = torch.randint(0, len(val_loader.dataset), (10,)).tolist()
    visualize_predictions(model, val_loader.dataset, vis_indices, title='ResNet18 Predictions', save_to='results/q2_resnet18_preds.png', device=device)

if __name__ == '__main__':
    main()
